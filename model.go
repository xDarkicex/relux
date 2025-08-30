package relux

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
)

// Network is a simple feedforward MLP built from dense layers.
type Network struct {
	layers    []layer.Layer
	inputSize int
	loss      loss.Loss
	rnd       *rand.Rand
}

// Predict runs a single forward pass through all layers.
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

// Fit trains the network on the provided dataset.
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

	// Validate input dimensions
	for i, x := range X {
		if len(x) != n.inputSize {
			return fmt.Errorf("training failed: sample %d has size %d, expected %d", i, len(x), n.inputSize)
		}
	}

	// Validate output dimensions
	expectedOutputSize := n.layers[len(n.layers)-1].(*layer.Dense).OutputSize()
	for i, y := range Y {
		if len(y) != expectedOutputSize {
			return fmt.Errorf("training failed: label %d has size %d, expected %d", i, len(y), expectedOutputSize)
		}
	}

	// Enhanced default config with Phase 5 features
	cfg := trainConfig{
		epochs:        10,
		batchSize:     32,
		learningRate:  0.01,
		shuffle:       true,
		verbose:       false,
		momentum:      0.0,  // No momentum by default
		lrDecay:       1.0,  // No decay by default
		lrDecaySteps:  1000, // Decay every 1000 epochs
		earlyStopping: 0,    // No early stopping by default
		gradClip:      5.0,  // Default gradient clipping
	}

	// Apply options
	for _, opt := range opts {
		opt(&cfg)
	}

	// Early stopping variables
	bestLoss := math.Inf(1)
	patience := 0
	currentLR := cfg.learningRate

	// Enhanced training loop with Phase 5 features
	for epoch := 0; epoch < cfg.epochs; epoch++ {
		var totalLoss float64
		sampleCount := 0

		// Learning rate decay
		if cfg.lrDecay < 1.0 && epoch > 0 && epoch%cfg.lrDecaySteps == 0 {
			currentLR *= cfg.lrDecay
			if cfg.verbose {
				fmt.Printf("Learning rate decayed to: %.6f\n", currentLR)
			}
		}

		// Create indices for shuffling
		indices := make([]int, len(X))
		for i := range indices {
			indices[i] = i
		}

		// Shuffle if requested
		if cfg.shuffle && n.rnd != nil {
			n.rnd.Shuffle(len(indices), func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		// Process batches
		for start := 0; start < len(indices); start += cfg.batchSize {
			end := start + cfg.batchSize
			if end > len(indices) {
				end = len(indices)
			}

			// Forward and backward for each sample in batch
			for i := start; i < end; i++ {
				idx := indices[i]

				// Forward through all layers
				out := X[idx]
				for _, layer := range n.layers {
					out = layer.Forward(out)
				}

				// Track loss for early stopping and monitoring
				lossValue := n.loss.Forward(out, Y[idx])
				totalLoss += lossValue
				sampleCount++

				// Enhanced backward pass with momentum and gradient clipping
				grad := n.loss.Backward(out, Y[idx])
				for j := len(n.layers) - 1; j >= 0; j-- {
					if dense, ok := n.layers[j].(*layer.Dense); ok {
						grad = dense.BackwardWithMomentum(grad, currentLR, cfg.momentum, cfg.gradClip)
					} else {
						grad = n.layers[j].Backward(grad, currentLR)
					}
				}
			}
		}

		// Early stopping check
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

		// Enhanced verbose logging with Phase 5 features
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

// Add getter methods for better encapsulation
func (n *Network) InputSize() int {
	return n.inputSize
}

func (n *Network) LayerCount() int {
	return len(n.layers)
}

func (n *Network) LossName() string {
	if n.loss == nil {
		return "none"
	}
	return n.loss.Name()
}

// GetLayers returns the internal layers (for serialization).
func (n *Network) GetLayers() []layer.Layer {
	return n.layers
}

// GetInputSize returns the input size (for serialization).
func (n *Network) GetInputSize() int {
	return n.inputSize
}

// SetInputSize sets the input size (for serialization).
func (n *Network) SetInputSize(size int) {
	n.inputSize = size
}

// SetLoss sets the loss function (for serialization).
func (n *Network) SetLoss(l loss.Loss) {
	n.loss = l
}

// SetLayers sets the layers (for serialization).
func (n *Network) SetLayers(layers []layer.Layer) {
	n.layers = layers
}
