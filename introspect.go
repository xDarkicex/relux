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
			total += len(dense.W) + len(dense.B)
		}
	}
	return total
}

// GetLayerWeights returns the weights (reshaped to [][]float64 for caller
// convenience) and biases for a specific layer. Returns copies.
func (n *Network) GetLayerWeights(layerIndex int) ([][]float32, []float32, error) {
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

	weights := make([][]float32, dense.Out())
	for j := 0; j < dense.Out(); j++ {
		row := make([]float32, dense.In())
		// dense.W is bfloat16 in memory; widen on the fly
		// for the introspect API which returns float32.
		for i, w := range dense.W[j*dense.In() : (j+1)*dense.In()] {
			row[i] = bf16ToFloat32(w)
		}
		weights[j] = row
	}
	biases := make([]float32, len(dense.B))
	for i, b := range dense.B {
		biases[i] = bf16ToFloat32(b)
	}

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

	expectedInput := n.inputSize
	for i, l := range n.layers {
		dense, ok := l.(*layer.Dense)
		if !ok {
			return fmt.Errorf("layer %d is not a Dense layer", i)
		}
		if len(dense.W) != expectedInput*dense.Out() {
			return fmt.Errorf("layer %d weight buffer has %d elements, expected %d",
				i, len(dense.W), expectedInput*dense.Out())
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
