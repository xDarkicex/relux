// Package optim provides gradient-based parameter optimizers for training.
//
// The package decouples the update rule (SGD, Adam, ...) from the layer that
// owns the parameters. A layer exposes its trainable parameters as []Param,
// and an Optimizer applies accumulated gradients to those parameters in place.
//
// Usage:
//
//	// Build a network.
//	net, _ := relux.NewNetwork(...)
//
//	// Train with a custom optimizer.
//	net.Fit(X, Y,
//	    relux.Epochs(1000),
//	    relux.LearningRate(0.001),
//	    relux.Optimizer(&optim.Adam{LR: 0.001}),
//	)
//
// Or stand-alone:
//
//	opt := &optim.SGD{LR: 0.01, Momentum: 0.9}
//	params := layer.Params()           // []optim.Param
//	opt.Step(params)
//
// The Stateful interface lets optimizers with internal state (momentum
// buffers, Adam moment estimates) be serialized alongside the model.
package optim

import "math"

// Param is a single trainable parameter together with its current gradient.
//
// Data and Grad are flat, length-equal slices. The caller is responsible for
// zeroing Grad after each Step if accumulating across samples; per-sample
// optimizers overwrite it on the next forward pass.
//
// Type contract (post-refactor):
//
//   - Data is bfloat16 (uint16 in memory; the bit pattern is the standard
//     IEEE 754 brain-float truncation: 8 bits of exponent matching
//     float32's range, 7 bits of mantissa). This is the active weight
//     that matmul / convolution operates on. The 16-bit precision gives
//     2x compute and 2x memory bandwidth on M-series GPUs and modern
//     accelerators; the bf16 mantissa loss is well within the noise
//     floor of transformer gradients.
//   - Grad is float32. Standard mixed-precision practice: gradients
//     accumulate in float32 to avoid the catastrophic cancellation
//     that bf16 accumulates. The optimizer reads Grad, updates Data
//     (downcasting the result to bf16).
//
// The Name field is used to key optimizer state (momentum buffers, Adam
// moments) across calls and across save/load.
type Param struct {
	Name string
	Data []uint16 // bfloat16
	Grad []float32
}

// Optimizer updates parameters based on their gradients.
//
// Implementations must mutate the Data slice of each Param in place.
// Grad is read but not cleared by Step.
type Optimizer interface {
	Step(params []Param) error
}

// State is a serializable snapshot of an optimizer's internal state.
//
// Buffers is keyed by parameter name. Implementations may attach any number
// of buffers (e.g. SGD uses one velocity buffer per param, Adam uses two).
// Step is the optimizer's internal timestep (used by Adam for bias
// correction; zero for stateless optimizers like vanilla SGD).
//
// Buffer values are float32. Adam's m and v must be float32 — bf16
// truncation of the optimizer's running averages (which are exponentially
// decaying sums of tiny gradients) causes quantization shocks on resume.
type State struct {
	Kind    string
	Buffers map[string][]float32
	Step    int
}

// Stateful is an Optimizer whose internal buffers can be exported and
// re-imported. relux's model serialization calls State / LoadState to
// include optimizer state in the .gob file.
type Stateful interface {
	Optimizer
	State() State
	LoadState(State) error
}

// ClipGradNorm rescales parameter gradients in place so their total L2 norm
// is at most maxNorm. This is the "global" gradient clip used by PyTorch
// (torch.nn.utils.clip_grad_norm_) and is applied across all parameters
// collectively, not per-layer.
//
// Returns the pre-clip L2 norm so callers can log it.
func ClipGradNorm(params []Param, maxNorm float32) float32 {
	var sumSq float32
	for _, p := range params {
		for _, g := range p.Grad {
			sumSq += g * g
		}
	}
	norm := float32(math.Sqrt(float64(sumSq)))
	if norm > maxNorm {
		scale := maxNorm / norm
		for _, p := range params {
			for i := range p.Grad {
				p.Grad[i] *= scale
			}
		}
	}
	return norm
}
