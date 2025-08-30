package serialize

import (
	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
)

// NetworkSnapshot represents a serializable version of a Network.
type NetworkSnapshot struct {
	InputSize int         `json:"inputSize"`
	Layers    []LayerData `json:"layers"`
	LossName  string      `json:"lossName"`
}

// LayerData represents serializable layer information.
type LayerData struct {
	Weights        [][]float64 `json:"weights"`
	Biases         []float64   `json:"biases"`
	ActivationName string      `json:"activation"`
}

// CreateSnapshot creates a snapshot from any network with the right methods.
func CreateSnapshot(getLayers func() []layer.Layer, getInputSize func() int, getLossName func() string) *NetworkSnapshot {
	layers := getLayers()

	snapshot := &NetworkSnapshot{
		InputSize: getInputSize(),
		LossName:  getLossName(),
		Layers:    make([]LayerData, len(layers)),
	}

	for i, l := range layers {
		if dense, ok := l.(*layer.Dense); ok {
			// Deep copy weights and biases
			weights := make([][]float64, len(dense.W))
			for j, row := range dense.W {
				weights[j] = make([]float64, len(row))
				copy(weights[j], row)
			}

			biases := make([]float64, len(dense.B))
			copy(biases, dense.B)

			snapshot.Layers[i] = LayerData{
				Weights:        weights,
				Biases:         biases,
				ActivationName: dense.Act.Name(),
			}
		}
	}

	return snapshot
}

// RestoreNetwork restores a network from a snapshot.
func RestoreNetwork(snapshot *NetworkSnapshot, setInputSize func(int), setLoss func(loss.Loss), setLayers func([]layer.Layer)) {
	setInputSize(snapshot.InputSize)
	setLoss(resolveLoss(snapshot.LossName))

	layers := make([]layer.Layer, len(snapshot.Layers))
	for i, layerData := range snapshot.Layers {
		dense := &layer.Dense{
			W:   layerData.Weights,
			B:   layerData.Biases,
			Act: resolveActivation(layerData.ActivationName),
		}

		// Initialize cache
		if len(layerData.Weights) > 0 {
			inputSize := len(layerData.Weights[0])
			outputSize := len(layerData.Weights)
			dense.InitializeCache(inputSize, outputSize)
		}

		layers[i] = dense
	}

	setLayers(layers)
}

// Helper functions (same as before)
func resolveActivation(name string) act.Activation {
	switch name {
	case "", "linear", "identity":
		return act.Identity()
	case "relu":
		return act.ReLU()
	case "sigmoid":
		return act.Sigmoid()
	case "tanh":
		return act.Tanh()
	default:
		return act.Identity()
	}
}

func resolveLoss(name string) loss.Loss {
	switch name {
	case "mse":
		return loss.MSE()
	case "bce":
		return loss.BCE()
	default:
		return loss.MSE()
	}
}
