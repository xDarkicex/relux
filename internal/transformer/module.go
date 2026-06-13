package transformer

import (
	"github.com/xDarkicex/relux/internal/compute"
	"github.com/xDarkicex/relux/internal/optim"
)

// Backend is the compute backend used by matmulBatched3D for
// hardware-accelerated matrix multiplication. Set by the
// relux.Transformer on construction (or left nil for pure
// Go). When non-nil, the bf16 weights are widened to f32
// once per matmul call and dispatched through the backend's
// MatMulFloat32, which routes through rnxa (MPS / Metal /
// CUDA) on supported platforms.
var Backend compute.ComputeBackend

// Mode controls the runtime behavior of a Module. Train enables
// gradient computation and (where applicable) dropout /
// stochastic ops. Inference disables them and routes through
// the KV-cache when the module is stateful.
//
// Modules that don't have mode-specific behavior (e.g. RMSNorm,
// MLP) can ignore the mode. Modules that do (MHA) read it via
// Mode() on the receiver.
type Mode int

const (
	// Train computes gradients and accumulates them into the
	// module's optim.Param.Grad slots. This is the default
	// after construction.
	Train Mode = iota
	// Inference disables gradient computation and runs the
	// forward pass through the KV-cache when the module has
	// one. Used by the autoregressive generation loop.
	Inference
)

// String reports the mode name. Used in error messages and the
// .relux v1 format's per-layer mode flag.
func (m Mode) String() string {
	switch m {
	case Train:
		return "train"
	case Inference:
		return "inference"
	default:
		return "unknown"
	}
}

// Module is the transformer-layer interface. It mirrors the
// existing internal/layer.Layer for the new 2D/3D/4D world:
//   - Forward takes a Tensor, returns a Tensor.
//   - Backward takes a gradient Tensor, returns the input-gradient
//     Tensor. It writes into the module's optim.Param.Grad slots
//     for any trainable parameters.
//   - Params returns the trainable parameters in the optim
//     package's format: Data []uint16 (bfloat16), Grad []float32.
//   - Mode() reports the current mode (Train/Inference). The
//     Engine mutator SetMode propagates this.
//
// All modules in this package are not safe for concurrent calls.
// The relux.Transformer type serialises via its own mutex.
type Module interface {
	// Forward computes the output Tensor from the input
	// Tensor. In Train mode it must cache whatever is needed
	// for Backward (typically the input, the pre-activation,
	// and any intermediate state). In Inference mode it
	// may skip the cache.
	Forward(x *Tensor) *Tensor

	// Backward computes the input-gradient Tensor from the
	// output-gradient Tensor. It accumulates into each
	// optim.Param.Grad slot of the module (the optimizer
	// reads these on the next Step). Returns the gradient
	// w.r.t. the input so the caller can chain back through
	// earlier modules.
	Backward(gradOut *Tensor) *Tensor

	// Params returns the trainable parameters in the
	// existing optim.Param contract: Data []uint16 (bf16),
	// Grad []float32. The active weight is widened from bf16
	// to f32 on the fly in Forward; the gradient accumulator
	// is f32; Backward adds to it.
	Params() []optim.Param

	// Mode returns the module's current mode.
	Mode() Mode

	// SetMode updates the module's mode. Called by the
	// Engine when the user calls relux.Transformer.SetMode.
	// The Engine is the single source of truth for mode
	// transitions; modules don't flip themselves.
	SetMode(Mode)
}

// BaseModule is the optional embedded struct for module
// implementations that just need a Mode field. Embedding
// gives them the SetMode and Mode methods for free.
type BaseModule struct {
	mode Mode
}

// Mode returns the current mode.
func (b *BaseModule) Mode() Mode { return b.mode }

// SetMode updates the mode. The default is Train (zero value
// of the int).
func (b *BaseModule) SetMode(m Mode) { b.mode = m }
