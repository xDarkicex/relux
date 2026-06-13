package serialize

import (
	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/layer"
	"github.com/xDarkicex/relux/internal/loss"
	"github.com/xDarkicex/relux/internal/optim"
)

// NetworkSnapshot is the serializable representation of a trained network.
// It also captures the optimizer state (momentum buffers, Adam moments) so a
// training run can be resumed from the exact point at which the snapshot
// was taken.
type NetworkSnapshot struct {
	InputSize int          `json:"inputSize"`
	Layers    []LayerData  `json:"layers"`
	LossName  string       `json:"lossName"`
	Optimizer *optim.State `json:"optimizer,omitempty"`
}

// LayerData holds a single Dense layer's parameters. Weights are stored as
// a flat row-major slice of length In*Out in bfloat16 (matching the live
// in-memory representation). The original shape lives in WeightShape so
// the network can be re-instantiated without inspecting adjacent layers.
type LayerData struct {
	Weights        []uint16 `json:"weights"` // bfloat16
	WeightShape    [2]int   `json:"weightShape"` // {In, Out}
	Biases         []uint16 `json:"biases"`       // bfloat16
	ActivationName string   `json:"activation"`
}

// CreateSnapshot builds a snapshot from a live network. The caller provides
// closures so the serializer does not depend on the relux.Network type
// (which would create an import cycle).
func CreateSnapshot(
	getLayers func() []layer.Layer,
	getInputSize func() int,
	getLossName func() string,
	getOptimizerState func() *optim.State,
) *NetworkSnapshot {
	layers := getLayers()
	snap := &NetworkSnapshot{
		InputSize: getInputSize(),
		LossName:  getLossName(),
		Layers:    make([]LayerData, 0, len(layers)),
	}

	for _, l := range layers {
		dense, ok := l.(*layer.Dense)
		if !ok {
			continue
		}
		wCopy := alloc.Uint16(len(dense.W))
		copy(wCopy, dense.W)
		bCopy := alloc.Uint16(len(dense.B))
		copy(bCopy, dense.B)
		snap.Layers = append(snap.Layers, LayerData{
			Weights:        wCopy,
			WeightShape:    [2]int{dense.In(), dense.Out()},
			Biases:         bCopy,
			ActivationName: dense.Act.Name(),
		})
	}

	if getOptimizerState != nil {
		snap.Optimizer = getOptimizerState()
	}
	return snap
}

// RestoreNetwork rebuilds a network's state from a snapshot. The caller
// provides setters that the serializer uses to install the restored state
// into the live network.
func RestoreNetwork(
	snapshot *NetworkSnapshot,
	setInputSize func(int),
	setLoss func(loss.Loss),
	setLayers func([]layer.Layer),
) {
	setInputSize(snapshot.InputSize)
	setLoss(resolveLoss(snapshot.LossName))

	layers := make([]layer.Layer, 0, len(snapshot.Layers))
	for i, ld := range snapshot.Layers {
		in, out := ld.WeightShape[0], ld.WeightShape[1]
		if in <= 0 || out <= 0 || len(ld.Weights) != in*out || len(ld.Biases) != out {
			// Skip malformed layer rather than crash; Validate() will surface it.
			continue
		}
		w := alloc.Uint16(len(ld.Weights))
		copy(w, ld.Weights)
		b := alloc.Uint16(len(ld.Biases))
		copy(b, ld.Biases)
		dense := &layer.Dense{
			W:   w,
			B:   b,
			Act: resolveActivation(ld.ActivationName),
		}
		dense.SetLayerID(i)
		dense.InitializeCache(in, out)
		layers = append(layers, dense)
	}
	setLayers(layers)
}

func resolveActivation(name string) act.Activation {
	switch name {
	case "", "linear", "identity":
		return act.Identity()
	case "relu":
		return act.ReLU()
	case "leaky_relu":
		return act.LeakyReLU(0.01)
	case "sigmoid":
		return act.Sigmoid()
	case "tanh":
		return act.Tanh()
	case "softmax":
		return act.Softmax()
	case "swish":
		return act.Swish()
	case "gelu":
		return act.GELU()
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
	case "categorical_crossentropy", "cce":
		return loss.CategoricalCrossentropy()
	case "sparse_categorical_crossentropy", "scce":
		return loss.SparseCategoricalCrossentropy()
	default:
		return loss.MSE()
	}
}
