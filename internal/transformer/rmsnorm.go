package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// RMSNorm is the Root Mean Square Layer Normalization used by
// LLaMA, Mistral, Qwen, and most modern LLMs. The
// pre-norm is computed as:
//
//	rstd = 1 / sqrt(mean(x^2, axis=-1, keepdim=True) + eps)
//	y = x * rstd * gamma
//
// There is no bias term (that's the difference from LayerNorm).
// gamma is the multiplicative weight, stored as bfloat16,
// initialized to 1 (so the layer is the identity at init).
//
// The forward operates on the last dim of an N-D tensor. For a
// 2-D input [batch, dModel], each row is normalized
// independently. For a 3-D input [batch, seq, dModel], each
// [b, s, :] row is normalized.
//
// Active dtype is float32 throughout. Weights widen from
// bf16 to f32 on read; gradients accumulate in f32.
type RMSNorm struct {
	BaseModule
	dModel int
	eps    float32

	// gamma is the multiplicative weight. Stored as
	// bfloat16 (per optim.Param.Data contract). The
	// active dtype is f32 (widened on read).
	gamma optim.Param

	// Forward cache. Set during Train-mode Forward, used by
	// Backward. Cleared on each Forward call.
	lastX    *Tensor
	lastRstd []float32 // length = product of all dims except the last
}

// NewRMSNorm constructs an RMSNorm with dModel features and
// the given eps. gamma is initialised to ones.
func NewRMSNorm(dModel int, eps float32) *RMSNorm {
	if dModel <= 0 {
		panic(fmt.Sprintf("NewRMSNorm: dModel=%d, must be > 0", dModel))
	}
	gammaData := alloc.Uint16(dModel)
	for i := range gammaData {
		gammaData[i] = BF16FromF32(1.0)
	}
	gammaGrad := alloc.Float32(dModel)
	return &RMSNorm{
		dModel: dModel,
		eps:    eps,
		gamma: optim.Param{
			Name: "rmsnorm.gamma",
			Data: gammaData,
			Grad: gammaGrad,
		},
	}
}

// Forward computes the normalized output in float32. The
// gamma weight is widened from bf16 to f32 on the fly.
func (r *RMSNorm) Forward(x *Tensor) *Tensor {
	if x.Rank() == 0 || x.shape[len(x.shape)-1] != r.dModel {
		panic(fmt.Sprintf("RMSNorm.Forward: last dim = %d, want %d",
			x.shape[len(x.shape)-1], r.dModel))
	}
	rows := x.Size() / r.dModel

	xData, _ := x.ToF32()
	gData := make([]float32, r.dModel)
	for i := range gData {
		gData[i] = F32FromBF16(r.gamma.Data[i])
	}

	out := ZerosF32(x.shape...)

	for row := 0; row < rows; row++ {
		base := row * r.dModel
		var sumSq float32
		for j := 0; j < r.dModel; j++ {
			v := xData[base+j]
			sumSq += v * v
		}
		meanSq := sumSq / float32(r.dModel)
		rstd := float32(1.0 / math.Sqrt(float64(meanSq + r.eps)))

		for j := 0; j < r.dModel; j++ {
			out.DataF32()[base+j] = xData[base+j] * rstd * gData[j]
		}
	}

	if r.Mode() == Train {
		r.lastX = &Tensor{shape: x.Shape(), dtype: Float32, f32: xData}
		if r.lastRstd == nil || len(r.lastRstd) != rows {
			r.lastRstd = alloc.Float32(rows)
		}
		for row := 0; row < rows; row++ {
			base := row * r.dModel
			var sumSq float32
			for j := 0; j < r.dModel; j++ {
				v := xData[base+j]
				sumSq += v * v
			}
			meanSq := sumSq / float32(r.dModel)
			r.lastRstd[row] = float32(1.0 / math.Sqrt(float64(meanSq + r.eps)))
		}
	}

	return out
}

// Backward computes the input gradient and accumulates into
// gamma.Grad (float32). The math is float32 throughout.
func (r *RMSNorm) Backward(gradOut *Tensor) *Tensor {
	if r.lastX == nil {
		panic("RMSNorm.Backward: Forward must be called first (Mode is not Train?)")
	}

	gOutData, _ := gradOut.ToF32()
	xData := r.lastX.DataF32()
	rows := gradOut.Size() / r.dModel

	gradIn := ZerosF32(gradOut.shape...)

	for row := 0; row < rows; row++ {
		base := row * r.dModel
		rstd := r.lastRstd[row]

		var dot float32
		for j := 0; j < r.dModel; j++ {
			dot += gOutData[base+j] * F32FromBF16(r.gamma.Data[j]) * xData[base+j]
		}

		scale := rstd * rstd * rstd * dot / float32(r.dModel) * 2
		for j := 0; j < r.dModel; j++ {
			gOutJ := gOutData[base+j]
			gJ := F32FromBF16(r.gamma.Data[j])
			xJ := xData[base+j]
			gradIn.DataF32()[base+j] = gOutJ*rstd*gJ - xJ*scale*xJ
			r.gamma.Grad[j] += gOutJ * xJ * rstd
		}
	}

	alloc.Free(r.lastX.f32)
	alloc.Free(r.lastRstd)
	r.lastX = nil
	r.lastRstd = nil

	return gradIn
}

// freeForwardCache releases the activation cache.
func (r *RMSNorm) freeForwardCache() {
	if r.lastX != nil {
		allocFreeTensor(r.lastX)
		r.lastX = nil
	}
	if r.lastRstd != nil {
		alloc.Free(r.lastRstd)
		r.lastRstd = nil
	}
}

// Params returns the gamma weight as an optim.Param.
func (r *RMSNorm) Params() []optim.Param { return []optim.Param{r.gamma} }

// GetParam returns a pointer to the underlying gamma Param.
func (r *RMSNorm) GetParam() *optim.Param { return &r.gamma }

// Eps returns the eps used by the layer.
func (r *RMSNorm) Eps() float32 { return r.eps }
