package relux

import (
	"fmt"
	"strings"

	"github.com/xDarkicex/relux/internal/layer"
)

// Summary returns a detailed string representation of the network architecture.
func (n *Network) Summary() string {
	if n == nil || len(n.layers) == 0 {
		return "Network: uninitialized"
	}

	var sb strings.Builder
	sb.WriteString("relux.Network Summary:\n")
	sb.WriteString("=====================\n")

	// Input layer
	sb.WriteString(fmt.Sprintf("Input: %d features\n", n.inputSize))

	// Hidden layers
	if len(n.layers) > 1 {
		sb.WriteString("Hidden Layers:\n")
		for i, l := range n.layers[:len(n.layers)-1] {
			if dense, ok := l.(*layer.Dense); ok {
				sb.WriteString(fmt.Sprintf("  Layer %d: %d units (%s)\n",
					i+1, dense.OutputSize(), dense.Act.Name()))
			}
		}
	}

	// Output layer
	if outputLayer, ok := n.layers[len(n.layers)-1].(*layer.Dense); ok {
		sb.WriteString(fmt.Sprintf("Output: %d units (%s)\n",
			outputLayer.OutputSize(), outputLayer.Act.Name()))
	}

	// Loss function
	sb.WriteString(fmt.Sprintf("Loss: %s\n", n.LossName()))

	// Parameter count
	sb.WriteString(fmt.Sprintf("Parameters: %d total\n", n.ParameterCount()))

	return sb.String()
}

// LayerSizes returns the output sizes of all layers in order.
func (n *Network) LayerSizes() []int {
	if n == nil || len(n.layers) == 0 {
		return []int{}
	}

	sizes := make([]int, len(n.layers))
	for i, l := range n.layers {
		if dense, ok := l.(*layer.Dense); ok {
			sizes[i] = dense.OutputSize()
		}
	}
	return sizes
}

// ParameterCount returns the total number of trainable parameters.
func (n *Network) ParameterCount() int {
	if n == nil || len(n.layers) == 0 {
		return 0
	}

	total := 0
	for _, l := range n.layers {
		if dense, ok := l.(*layer.Dense); ok {
			// Weights + biases
			weights := len(dense.W) * len(dense.W[0])
			biases := len(dense.B)
			total += weights + biases
		}
	}
	return total
}

// GetLayerWeights returns the weights and biases for a specific layer.
// Returns copies to prevent accidental modification.
func (n *Network) GetLayerWeights(layerIndex int) ([][]float64, []float64, error) {
	if n == nil || len(n.layers) == 0 {
		return nil, nil, fmt.Errorf("network not initialized")
	}
	if layerIndex < 0 || layerIndex >= len(n.layers) {
		return nil, nil, fmt.Errorf("layer index %d out of range [0, %d)", layerIndex, len(n.layers))
	}

	dense, ok := n.layers[layerIndex].(*layer.Dense)
	if !ok {
		return nil, nil, fmt.Errorf("layer %d is not a Dense layer", layerIndex)
	}

	// Create deep copies
	weights := make([][]float64, len(dense.W))
	for i, row := range dense.W {
		weights[i] = make([]float64, len(row))
		copy(weights[i], row)
	}

	biases := make([]float64, len(dense.B))
	copy(biases, dense.B)

	return weights, biases, nil
}

// Validate performs comprehensive validation of the network structure.
func (n *Network) Validate() error {
	if n == nil {
		return fmt.Errorf("network is nil")
	}
	if len(n.layers) == 0 {
		return fmt.Errorf("network has no layers")
	}
	if n.inputSize <= 0 {
		return fmt.Errorf("invalid input size: %d", n.inputSize)
	}
	if n.loss == nil {
		return fmt.Errorf("network has no loss function")
	}

	// Validate layer connectivity
	expectedInput := n.inputSize
	for i, l := range n.layers {
		dense, ok := l.(*layer.Dense)
		if !ok {
			return fmt.Errorf("layer %d is not a Dense layer", i)
		}

		if len(dense.W) == 0 || len(dense.W[0]) != expectedInput {
			return fmt.Errorf("layer %d has invalid weight dimensions", i)
		}

		expectedInput = dense.OutputSize()
	}

	return nil
}

// Architecture returns a compact string representation of the network architecture.
func (n *Network) Architecture() string {
	if n == nil || len(n.layers) == 0 {
		return "uninitialized"
	}

	var parts []string
	parts = append(parts, fmt.Sprintf("%d", n.inputSize))

	for _, l := range n.layers {
		if dense, ok := l.(*layer.Dense); ok {
			parts = append(parts, fmt.Sprintf("%d(%s)", dense.OutputSize(), dense.Act.Name()))
		}
	}

	return strings.Join(parts, " -> ")
}
