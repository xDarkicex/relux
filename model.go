package relux

import (
	"fmt"
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

	// Default training config
	cfg := trainConfig{
		epochs:       10,
		batchSize:    32,
		learningRate: 0.01,
		shuffle:      true,
		verbose:      false, // Added verbose default
	}

	// Apply options
	for _, opt := range opts {
		opt(&cfg)
	}

	// Training loop with verbose monitoring
	for epoch := 0; epoch < cfg.epochs; epoch++ {
		var totalLoss float64
		sampleCount := 0

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

				// Accumulate loss for monitoring
				if cfg.verbose {
					loss := n.loss.Forward(out, Y[idx])
					totalLoss += loss
					sampleCount++
				}

				// Backward pass
				grad := n.loss.Backward(out, Y[idx])
				for j := len(n.layers) - 1; j >= 0; j-- {
					grad = n.layers[j].Backward(grad, cfg.learningRate)
				}
			}
		}

		// Verbose logging every 100 epochs or last epoch
		if cfg.verbose && ((epoch+1)%100 == 0 || epoch == cfg.epochs-1) {
			avgLoss := totalLoss / float64(sampleCount)
			fmt.Printf("Epoch %d/%d: avg_loss=%.6f\n", epoch+1, cfg.epochs, avgLoss)
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
