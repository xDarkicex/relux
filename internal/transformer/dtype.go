// Package transformer provides the building blocks for the
// relux.Transformer LLM: a Tensor type with bfloat16 + float32
// active buffers, an RMSNorm / LayerNorm / RoPE / Embedding / MHA
// / Transformer block / Sampler / Generate loop, and a KV-cache.
//
// The package is decoupled from the existing internal/layer
// package (which is 1D, float64, MLP-oriented). The Module
// interface here is the new "2D/3D/4D, mixed precision, batched"
// analogue. The two coexist: the existing relux.Network uses
// Layer/Dense for MLPs; the new relux.Transformer uses
// Module/transformer primitives for LLMs.
//
// Mixed precision: the master weight is float32 (the optim/ package
// contract — float64 Param.Data + Param.Grad is reused as float32
// here for the module's public face, see module.go for the bridge).
// The active buffer on the GPU is bfloat16. The bfloat16 ↔ float32
// conversion happens in dtype.go.
package transformer

// DType identifies the active numeric type of a Tensor.
type DType int

const (
	// Float32 is IEEE 754 single precision. The default active
	// type for tensors holding the master weight.
	Float32 DType = iota
	// BFloat16 is the "brain float" 16-bit format: 8 bits of
	// exponent (same range as float32), 7 bits of mantissa.
	// Used for activations, KV-cache, and the active weight on
	// the GPU. Saves 2x memory vs float32; the lossy mantissa
	// is fine for neural networks because the exponent range
	// (the dynamic range) is preserved.
	BFloat16
	// Float64 is IEEE 754 double precision. Used for gradients
	// and (rarely) the master weight when extra precision is
	// required.
	Float64
)

// String reports the dtype as a stable identifier — used in the
// .relux v1 binary format header.
func (d DType) String() string {
	switch d {
	case Float32:
		return "f32"
	case BFloat16:
		return "bf16"
	case Float64:
		return "f64"
	default:
		return "unknown"
	}
}

// BytesPerElem reports the byte size of one element of d.
func (d DType) BytesPerElem() int {
	switch d {
	case Float32:
		return 4
	case BFloat16:
		return 2
	case Float64:
		return 8
	default:
		return 0
	}
}
