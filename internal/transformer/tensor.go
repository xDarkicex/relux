package transformer

import (
	"fmt"

	"github.com/xDarkicex/relux/internal/alloc"
)

// Tensor is the transformer's n-dimensional array. Exactly one of
// f32, bf16, or f64 is non-nil — the "active" dtype. The master
// weight and the gradient live in the optim package's Param
// struct (float64 Data + float64 Grad); the Tensor is the
// "active" view that gets passed between modules during
// forward/backward.
//
// Storage is off-heap via the alloc package. A Tensor is not
// safe to share across goroutines; the Module interface
// serialises calls via the engine wrapper.
//
// Conventions:
//   - Row-major: for shape [batch, seq, dModel], the underlying
//     slice is indexed as [b*seq*dModel + s*dModel + d].
//   - Total element count = product(shape). All three data slices
//     (when present) have exactly that many elements.
type Tensor struct {
	shape []int
	dtype DType

	// Exactly one of these is non-nil, matching dtype.
	f32  []float32
	bf16 []uint16
	f64  []float64
}

// NewTensor wraps a float32 slice as a float32 Tensor. The
// shape is taken from `shape...`; len(data) must equal the
// product. The data slice is borrowed (not copied); the caller
// must not modify it after handing it off.
func NewTensor(data []float32, shape ...int) *Tensor {
	size := shapeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("transformer.NewTensor: len(data)=%d, want %d for shape %v", len(data), size, shape))
	}
	return &Tensor{shape: append([]int(nil), shape...), dtype: Float32, f32: data}
}

// NewTensorFromBF16 wraps a bfloat16 (uint16) slice as a bfloat16
// Tensor. Same conventions as NewTensor.
func NewTensorFromBF16(data []uint16, shape ...int) *Tensor {
	size := shapeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("transformer.NewTensorFromBF16: len(data)=%d, want %d for shape %v", len(data), size, shape))
	}
	return &Tensor{shape: append([]int(nil), shape...), dtype: BFloat16, bf16: data}
}

// NewTensorFromF64 wraps a float64 slice as a float64 Tensor.
// Used for gradient tensors that the master-weight optim/
// pipeline produces.
func NewTensorFromF64(data []float64, shape ...int) *Tensor {
	size := shapeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("transformer.NewTensorFromF64: len(data)=%d, want %d for shape %v", len(data), size, shape))
	}
	return &Tensor{shape: append([]int(nil), shape...), dtype: Float64, f64: data}
}

// ZerosF32 returns a zero-filled float32 Tensor with the given
// shape, backed by alloc.Float32.
func ZerosF32(shape ...int) *Tensor {
	size := shapeSize(shape)
	return &Tensor{shape: append([]int(nil), shape...), dtype: Float32, f32: alloc.Float32(size)}
}

// ZerosBF16 returns a zero-filled bfloat16 Tensor with the given
// shape, backed by alloc.Uint16.
func ZerosBF16(shape ...int) *Tensor {
	size := shapeSize(shape)
	return &Tensor{shape: append([]int(nil), shape...), dtype: BFloat16, bf16: alloc.Uint16(size)}
}

// ZerosF64 returns a zero-filled float64 Tensor with the given
// shape, backed by alloc.Float64.
func ZerosF64(shape ...int) *Tensor {
	size := shapeSize(shape)
	return &Tensor{shape: append([]int(nil), shape...), dtype: Float64, f64: alloc.Float64(size)}
}

// Shape returns a copy of the shape slice; mutating it does not
// affect the Tensor.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int { return len(t.shape) }

// Size returns the total element count (product of shape).
func (t *Tensor) Size() int { return shapeSize(t.shape) }

// DType reports the active dtype.
func (t *Tensor) DType() DType { return t.dtype }

// DataF32 returns the underlying float32 slice. Panics if the
// Tensor is not float32. The slice is borrowed.
func (t *Tensor) DataF32() []float32 {
	if t.dtype != Float32 {
		panic(fmt.Sprintf("Tensor.DataF32 called on dtype=%s", t.dtype))
	}
	return t.f32
}

// DataBF16 returns the underlying bfloat16 (uint16) slice.
// Panics if the Tensor is not bfloat16. The slice is borrowed.
func (t *Tensor) DataBF16() []uint16 {
	if t.dtype != BFloat16 {
		panic(fmt.Sprintf("Tensor.DataBF16 called on dtype=%s", t.dtype))
	}
	return t.bf16
}

// DataF64 returns the underlying float64 slice. Panics if the
// Tensor is not float64. The slice is borrowed.
func (t *Tensor) DataF64() []float64 {
	if t.dtype != Float64 {
		panic(fmt.Sprintf("Tensor.DataF64 called on dtype=%s", t.dtype))
	}
	return t.f64
}

// ToF32 returns a float32 view of the Tensor. If the Tensor is
// already float32, the underlying slice is returned (no copy).
// If bfloat16, the slice is widened to a fresh float32 (alloc.
// Float32, off-heap). If float64, the slice is narrowed to a
// fresh float32 (also off-heap).
func (t *Tensor) ToF32() ([]float32, []int) {
	switch t.dtype {
	case Float32:
		return t.f32, t.Shape()
	case BFloat16:
		out := alloc.Float32(len(t.bf16))
		for i, v := range t.bf16 {
			out[i] = F32FromBF16(v)
		}
		return out, t.Shape()
	case Float64:
		out := alloc.Float32(len(t.f64))
		for i, v := range t.f64 {
			out[i] = float32(v)
		}
		return out, t.Shape()
	default:
		panic("Tensor.ToF32: unknown dtype")
	}
}

// ToBF16 returns a bfloat16 (uint16) view of the Tensor. If the
// Tensor is already bfloat16, the underlying slice is returned.
// If float32, the slice is cast to a fresh bfloat16 (alloc.Uint16).
// If float64, the slice is narrowed to float32 then cast to
// bfloat16.
func (t *Tensor) ToBF16() ([]uint16, []int) {
	switch t.dtype {
	case BFloat16:
		return t.bf16, t.Shape()
	case Float32:
		return BF16SliceFromF32(t.f32), t.Shape()
	case Float64:
		return BF16SliceFromF64(t.f64), t.Shape()
	default:
		panic("Tensor.ToBF16: unknown dtype")
	}
}

// ToF64 returns a float64 view of the Tensor. Always copies (no
// zero-copy path) because the master weight in the optim package
// is float64 and we want the convert-once-at-the-boundary
// semantic. Backed by alloc.Float64.
func (t *Tensor) ToF64() ([]float64, []int) {
	switch t.dtype {
	case Float64:
		return t.f64, t.Shape()
	case Float32:
		out := alloc.Float64(len(t.f32))
		for i, v := range t.f32 {
			out[i] = float64(v)
		}
		return out, t.Shape()
	case BFloat16:
		return F64SliceFromBF16(t.bf16), t.Shape()
	default:
		panic("Tensor.ToF64: unknown dtype")
	}
}

// Clone returns a deep copy of the Tensor. The new Tensor's data
// slice is fresh off-heap allocation; mutations don't propagate.
func (t *Tensor) Clone() *Tensor {
	out := &Tensor{shape: t.Shape(), dtype: t.dtype}
	switch t.dtype {
	case Float32:
		out.f32 = alloc.Float32(len(t.f32))
		copy(out.f32, t.f32)
	case BFloat16:
		out.bf16 = alloc.Uint16(len(t.bf16))
		copy(out.bf16, t.bf16)
	case Float64:
		out.f64 = alloc.Float64(len(t.f64))
		copy(out.f64, t.f64)
	}
	return out
}

// Reshape returns a Tensor with the same data but a different
// shape. The element count must match.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	want := shapeSize(shape)
	if want != t.Size() {
		panic(fmt.Sprintf("Tensor.Reshape: shape %v has %d elements, tensor has %d", shape, want, t.Size()))
	}
	return &Tensor{shape: append([]int(nil), shape...), dtype: t.dtype, f32: t.f32, bf16: t.bf16, f64: t.f64}
}

// shapeSize returns the product of a shape slice. A zero-length
// shape (scalar) is treated as size 1, matching PyTorch's
// numel() convention.
func shapeSize(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
