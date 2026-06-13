package relux

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
	"github.com/xDarkicex/relux/internal/optim"
)

// Network is a simple feedforward MLP built from dense layers.
//
// The optimizer field, when non-nil, owns the per-parameter update rule.
// The Fit loop orchestrates forward / backward / step; it does not
// implement momentum, Adam, or any other update algorithm itself.
type Network struct {
	layers    []layer.Layer
	inputSize int
	loss      loss.Loss
	rnd       *rand.Rand

	// optimizer persists across Fit calls and save/load round-trips.
	optimizer optim.Optimizer
	// restoredState is set by Load() so a snapshot resumes training with
	// the original momentum / Adam buffers intact. Fit consumes and clears
	// it on the first invocation.
	restoredState *optim.State
}

// Predict runs a single forward pass.
func (n *Network) Predict(x []float64) ([]float64, error) {
	if n == nil || len(n.layers) == 0 {
		return nil, fmt.Errorf("network not initialized")
	}
	if len(x) != n.inputSize {
		return nil, fmt.Errorf("input size %d does not match expected %d", len(x), n.inputSize)
	}
	out := x
	for _, l := range n.layers {
		out = l.Forward(out)
	}
	return out, nil
}

// collectParams concatenates Params() across all layers into a single slice.
// The returned slice is rebuilt on every call; safe to pass to optim.Step.
func (n *Network) collectParams() []optim.Param {
	var all []optim.Param
	for _, l := range n.layers {
		all = append(all, l.Params()...)
	}
	return all
}

// optimizerState returns the current optimizer's serialized state, or nil
// if the optimizer does not implement optim.Stateful.
func (n *Network) optimizerState() *optim.State {
	if n.optimizer == nil {
		return nil
	}
	if s, ok := n.optimizer.(optim.Stateful); ok {
		state := s.State()
		return &state
	}
	return nil
}

// loadOptimizerState restores a previously serialized optimizer state.
// No-op if either side is missing.
func (n *Network) loadOptimizerState(state *optim.State) {
	if state == nil || n.optimizer == nil {
		return
	}
	if s, ok := n.optimizer.(optim.Stateful); ok {
		_ = s.LoadState(*state)
	}
}

// Fit trains the network. The training loop:
//
//  1. Sets the optimizer (explicit > momentum-derived SGD > new SGD).
//  2. Restores any optimizer state from a prior Load().
//  3. Per epoch: optionally decay LR, shuffle, then walk samples.
//  4. Per sample: forward, loss, backward, optional global grad clip, step.
//  5. After each epoch: early-stopping check, verbose log.
func (n *Network) Fit(X, Y [][]float64, opts ...TrainOption) error {
	if n == nil {
		return fmt.Errorf("network is nil")
	}
	if len(n.layers) == 0 {
		return fmt.Errorf("network not properly initialized: no layers")
	}
	if len(X) == 0 {
		return fmt.Errorf("training failed: empty dataset")
	}
	if len(X) != len(Y) {
		return fmt.Errorf("training failed: dataset size mismatch X(%d) != Y(%d)", len(X), len(Y))
	}
	for i, x := range X {
		if len(x) != n.inputSize {
			return fmt.Errorf("training failed: sample %d has size %d, expected %d", i, len(x), n.inputSize)
		}
	}
	expectedOutputSize := n.layers[len(n.layers)-1].(*layer.Dense).OutputSize()
	for i, y := range Y {
		if len(y) != expectedOutputSize {
			return fmt.Errorf("training failed: label %d has size %d, expected %d", i, len(y), expectedOutputSize)
		}
	}

	cfg := trainConfig{
		epochs:        10,
		batchSize:     32,
		learningRate:  0.01,
		shuffle:       true,
		verbose:       false,
		momentum:      0.0,
		lrDecay:       1.0,
		lrDecaySteps:  1000,
		earlyStopping: 0,
		gradClip:      0.0,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Pick the optimizer, in priority order:
	//  1. explicit Optimizer() option,
	//  2. an optimizer already installed on the Network (preserves Adam
	//     moments / SGD velocity across Fit calls and save/load round-trips),
	//  3. a fresh SGD derived from the legacy Momentum() train option.
	if cfg.optimizer != nil {
		n.optimizer = cfg.optimizer
	} else if n.optimizer == nil {
		n.optimizer = &optim.SGD{
			LR:       float32(cfg.learningRate),
			Momentum: float32(cfg.momentum),
		}
	}

	if n.restoredState != nil {
		n.loadOptimizerState(n.restoredState)
		n.restoredState = nil
	}

	// Cache the param slice once — its length and names are stable for the
	// lifetime of the network; only the underlying Data/Grad slices change.
	allParams := n.collectParams()

	bestLoss := math.Inf(1)
	patience := 0
	currentLR := cfg.learningRate

	for epoch := 0; epoch < cfg.epochs; epoch++ {
		var totalLoss float64
		sampleCount := 0

		if cfg.lrDecay < 1.0 && epoch > 0 && epoch%cfg.lrDecaySteps == 0 {
			currentLR *= cfg.lrDecay
			if cfg.verbose {
				fmt.Printf("Learning rate decayed to: %.6f\n", currentLR)
			}
		}

		indices := make([]int, len(X))
		for i := range indices {
			indices[i] = i
		}
		if cfg.shuffle && n.rnd != nil {
			n.rnd.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		for start := 0; start < len(indices); start += cfg.batchSize {
			end := start + cfg.batchSize
			if end > len(indices) {
				end = len(indices)
			}

			for i := start; i < end; i++ {
				idx := indices[i]

				out := X[idx]
				for _, l := range n.layers {
					out = l.Forward(out)
				}

				lossValue := n.loss.Forward(out, Y[idx])
				totalLoss += lossValue
				sampleCount++

				grad := n.loss.Backward(out, Y[idx])
				for j := len(n.layers) - 1; j >= 0; j-- {
					grad = n.layers[j].Backward(grad)
				}

				if cfg.gradClip > 0 {
					optim.ClipGradNorm(allParams, float32(cfg.gradClip))
				}

				if err := n.optimizer.Step(allParams); err != nil {
					return fmt.Errorf("training failed: optimizer step: %w", err)
				}
			}
		}

		if cfg.earlyStopping > 0 {
			avgLoss := totalLoss / float64(sampleCount)
			if avgLoss < bestLoss {
				bestLoss = avgLoss
				patience = 0
			} else {
				patience++
				if patience >= cfg.earlyStopping {
					if cfg.verbose {
						fmt.Printf("Early stopping at epoch %d (best_loss=%.6f)\n", epoch+1, bestLoss)
					}
					break
				}
			}
		}

		if cfg.verbose && ((epoch+1)%100 == 0 || epoch == cfg.epochs-1) {
			avgLoss := totalLoss / float64(sampleCount)
			fmt.Printf("Epoch %d/%d: loss=%.6f, lr=%.6f", epoch+1, cfg.epochs, avgLoss, currentLR)
			if cfg.earlyStopping > 0 {
				fmt.Printf(", patience=%d/%d", patience, cfg.earlyStopping)
			}
			fmt.Println()
		}
	}

	return nil
}

func (n *Network) InputSize() int  { return n.inputSize }
func (n *Network) LayerCount() int { return len(n.layers) }

func (n *Network) LossName() string {
	if n.loss == nil {
		return "none"
	}
	return n.loss.Name()
}

// GetLayers returns the internal layers (for serialization).
func (n *Network) GetLayers() []layer.Layer { return n.layers }

// GetInputSize returns the input size (for serialization).
func (n *Network) GetInputSize() int { return n.inputSize }

// SetInputSize sets the input size (for serialization).
func (n *Network) SetInputSize(size int) { n.inputSize = size }

// SetLoss sets the loss function (for serialization).
func (n *Network) SetLoss(l loss.Loss) { n.loss = l }

// SetLayers sets the layers (for serialization).
func (n *Network) SetLayers(layers []layer.Layer) { n.layers = layers }

// SetOptimizer installs a custom optimizer and clears any restored state.
// Use this after Load() if you want to swap algorithms.
func (n *Network) SetOptimizer(o optim.Optimizer) { n.optimizer = o }

// OptimizerForTest returns the currently installed optimizer. Test-only
// helper; do not use in production code.
func (n *Network) OptimizerForTest() optim.Optimizer { return n.optimizer }
