package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// Linear is a single y = x @ W + b layer (no activation,
// no two-layer FFN). Used for the language model head and
// for any place that needs a single linear projection
// (not a FFN block).
//
// W and b are stored as bfloat16. The matmul happens in
// float32: each bf16 weight is widened to f32 on the fly,
// multiplied with the f32 input, and the result accumulated
// in f32. Gradients accumulate in float32.
type Linear struct {
	BaseModule
	inDim  int
	outDim int

	W optim.Param
	b optim.Param

	lastX []float32
}

// NewLinear constructs a Linear with He init on the weight.
func NewLinear(inDim, outDim int) *Linear {
	if inDim <= 0 || outDim <= 0 {
		panic(fmt.Sprintf("NewLinear: inDim=%d, outDim=%d, both must be > 0", inDim, outDim))
	}
	stddev := float32(1.0 / math.Sqrt(float64(inDim)))
	wF32 := alloc.Float32(inDim * outDim)
	for i := range wF32 {
		r := float32(((i*1103515245 + 12345) & 0x7fffffff) % 1000) / 500.0 - 1.0
		wF32[i] = r * stddev
	}
	wData := alloc.Uint16(inDim * outDim)
	for i, x := range wF32 {
		wData[i] = BF16FromF32(x)
	}
	return &Linear{
		inDim:  inDim,
		outDim: outDim,
		W: optim.Param{
			Name: "linear.W",
			Data: wData,
			Grad: alloc.Float32(inDim * outDim),
		},
		b: optim.Param{
			Name: "linear.b",
			Data: alloc.Uint16(outDim),
			Grad: alloc.Float32(outDim),
		},
	}
}

// Forward computes y = x @ W + b. Input shape [..., inDim]
// -> output shape [..., outDim]. Weights widen from bf16
// to f32 on the fly.
func (l *Linear) Forward(x *Tensor) *Tensor {
	if x.Rank() == 0 || x.shape[len(x.shape)-1] != l.inDim {
		panic(fmt.Sprintf("Linear.Forward: last dim = %d, want %d",
			x.shape[len(x.shape)-1], l.inDim))
	}
	batch := 1
	for i := 0; i < x.Rank()-1; i++ {
		batch *= x.shape[i]
	}
	xData, _ := x.ToF32()
	y := ZerosF32(append(append([]int{}, x.shape[:x.Rank()-1]...), l.outDim)...)
	// matmulBatched3D takes an optim.Param; for the linear
	// layer we widen on the fly.
	wData := make([]float32, len(l.W.Data))
	for i, w := range l.W.Data {
		wData[i] = F32FromBF16(w)
	}
	for i := 0; i < batch; i++ {
		for j := 0; j < l.outDim; j++ {
			var sum float32
			for k := 0; k < l.inDim; k++ {
				sum += xData[i*l.inDim+k] * wData[k*l.outDim+j]
			}
			y.DataF32()[i*l.outDim+j] = sum
		}
	}
	bData := make([]float32, l.outDim)
	for i, b := range l.b.Data {
		bData[i] = F32FromBF16(b)
	}
	for i := range y.DataF32() {
		y.DataF32()[i] += bData[i%l.outDim]
	}
	if l.Mode() == Train {
		l.lastX = xData
	} else {
		alloc.Free(xData)
	}
	return y
}

// Backward accumulates W.Grad and b.Grad (float32) and
// returns the input gradient.
func (l *Linear) Backward(gradOut *Tensor) *Tensor {
	if l.lastX == nil {
		panic("Linear.Backward: Forward must be called first (Mode is not Train?)")
	}
	gOut, _ := gradOut.ToF32()
	batch := 1
	for i := 0; i < gradOut.Rank()-1; i++ {
		batch *= gradOut.shape[i]
	}
	wData := make([]float32, len(l.W.Data))
	for i, w := range l.W.Data {
		wData[i] = F32FromBF16(w)
	}
	xData := l.lastX

	// dL/dW += x^T @ gradOut.
	for r := 0; r < l.inDim; r++ {
		for j := 0; j < l.outDim; j++ {
			var sum float32
			for i := 0; i < batch; i++ {
				sum += xData[i*l.inDim+r] * gOut[i*l.outDim+j]
			}
			l.W.Grad[r*l.outDim+j] += sum
		}
	}
	// dL/db += sum_i gradOut[i, :]
	for j := 0; j < l.outDim; j++ {
		var sum float32
		for i := 0; i < batch; i++ {
			sum += gOut[i*l.outDim+j]
		}
		l.b.Grad[j] += sum
	}
	// dL/dX = gradOut @ W^T.
	gradIn := ZerosF32(append(append([]int{}, gradOut.shape[:gradOut.Rank()-1]...), l.inDim)...)
	for i := 0; i < batch; i++ {
		for r := 0; r < l.inDim; r++ {
			var sum float32
			for j := 0; j < l.outDim; j++ {
				sum += gOut[i*l.outDim+j] * wData[j*l.inDim+r]
			}
			gradIn.DataF32()[i*l.inDim+r] = sum
		}
	}
	alloc.Free(l.lastX)
	l.lastX = nil
	return gradIn
}

// Params returns W, b.
func (l *Linear) Params() []optim.Param { return []optim.Param{l.W, l.b} }

// WParam returns a pointer to the W weight.
func (l *Linear) WParam() *optim.Param { return &l.W }

// BParam returns a pointer to the b bias.
func (l *Linear) BParam() *optim.Param { return &l.b }
