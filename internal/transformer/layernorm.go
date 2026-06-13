package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// LayerNorm is the standard Layer Normalization. The pre-norm
// is computed as:
//
//	mean = mean(x, axis=-1, keepdim=True)
//	var  = mean((x - mean)^2, axis=-1, keepdim=True)
//	y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// gamma is the multiplicative scale (init 1, stored as bf16),
// beta is the additive shift (init 0, stored as bf16). The
// init is the identity function — at init the layer is y = x.
type LayerNorm struct {
	BaseModule
	dModel int
	eps    float32

	gamma optim.Param
	beta  optim.Param

	// Forward cache. Cleared after Backward.
	lastX    *Tensor
	lastMean []float32
	lastVar  []float32
	lastStd  []float32
}

// NewLayerNorm constructs a LayerNorm with dModel features and
// the given eps. gamma is initialised to 1, beta to 0.
func NewLayerNorm(dModel int, eps float32) *LayerNorm {
	if dModel <= 0 {
		panic(fmt.Sprintf("NewLayerNorm: dModel=%d, must be > 0", dModel))
	}
	gammaData := alloc.Uint16(dModel)
	betaData := alloc.Uint16(dModel)
	for i := range gammaData {
		gammaData[i] = BF16FromF32(1.0)
		// betaData[i] is 0 from zero-init
	}
	return &LayerNorm{
		dModel: dModel,
		eps:    eps,
		gamma: optim.Param{
			Name: "layernorm.gamma",
			Data: gammaData,
			Grad: alloc.Float32(dModel),
		},
		beta: optim.Param{
			Name: "layernorm.beta",
			Data: betaData,
			Grad: alloc.Float32(dModel),
		},
	}
}

// Forward computes the normalized output. Same per-row
// convention as RMSNorm. Weights widen from bf16 to f32.
func (l *LayerNorm) Forward(x *Tensor) *Tensor {
	if x.Rank() == 0 || x.shape[len(x.shape)-1] != l.dModel {
		panic(fmt.Sprintf("LayerNorm.Forward: last dim = %d, want %d",
			x.shape[len(x.shape)-1], l.dModel))
	}
	rows := x.Size() / l.dModel

	xData, _ := x.ToF32()
	gData := make([]float32, l.dModel)
	bData := make([]float32, l.dModel)
	for i := range gData {
		gData[i] = F32FromBF16(l.gamma.Data[i])
		bData[i] = F32FromBF16(l.beta.Data[i])
	}

	out := ZerosF32(x.shape...)

	if l.lastMean == nil || len(l.lastMean) != rows {
		l.lastMean = alloc.Float32(rows)
		l.lastVar = alloc.Float32(rows)
		l.lastStd = alloc.Float32(rows)
	}

	for row := 0; row < rows; row++ {
		base := row * l.dModel

		var sum float32
		for j := 0; j < l.dModel; j++ {
			sum += xData[base+j]
		}
		mean := sum / float32(l.dModel)
		l.lastMean[row] = mean

		var vsum float32
		for j := 0; j < l.dModel; j++ {
			d := xData[base+j] - mean
			vsum += d * d
		}
		var_ := vsum / float32(l.dModel)
		l.lastVar[row] = var_
		std := float32(math.Sqrt(float64(var_ + l.eps)))
		l.lastStd[row] = std

		invStd := 1.0 / std
		for j := 0; j < l.dModel; j++ {
			h := (xData[base+j] - mean) * invStd
			out.DataF32()[base+j] = h*gData[j] + bData[j]
		}
	}

	if l.Mode() == Train {
		l.lastX = &Tensor{shape: x.Shape(), dtype: Float32, f32: xData}
	} else {
		alloc.Free(xData)
	}

	return out
}

// Backward computes the input gradient and accumulates into
// gamma.Grad and beta.Grad (float32).
func (l *LayerNorm) Backward(gradOut *Tensor) *Tensor {
	if l.lastX == nil {
		panic("LayerNorm.Backward: Forward must be called first (Mode is not Train?)")
	}

	gOutData, _ := gradOut.ToF32()
	xData := l.lastX.DataF32()
	rows := gradOut.Size() / l.dModel

	gradIn := ZerosF32(gradOut.shape...)

	for row := 0; row < rows; row++ {
		base := row * l.dModel
		mean := l.lastMean[row]
		std := l.lastStd[row]
		invStd := 1.0 / std

		var sumDlDhGamma, sumDlDhGammaH float32
		for j := 0; j < l.dModel; j++ {
			h := (xData[base+j] - mean) * invStd
			gJ := F32FromBF16(l.gamma.Data[j])
			dlDhJ := gOutData[base+j]
			dlDhGammaJ := dlDhJ * gJ
			sumDlDhGamma += dlDhGammaJ
			sumDlDhGammaH += dlDhGammaJ * h
			l.gamma.Grad[j] += dlDhGammaJ * h
			l.beta.Grad[j] += dlDhJ
		}

		meanDlDhGamma := sumDlDhGamma / float32(l.dModel)
		meanDlDhGammaH := sumDlDhGammaH / float32(l.dModel)
		for j := 0; j < l.dModel; j++ {
			h := (xData[base+j] - mean) * invStd
			gJ := F32FromBF16(l.gamma.Data[j])
			dlDhJ := gOutData[base+j]
			gradIn.DataF32()[base+j] = invStd * (dlDhJ*gJ - meanDlDhGamma - h*meanDlDhGammaH)
		}
	}

	alloc.Free(l.lastX.f32)
	l.lastX = nil

	return gradIn
}

// Params returns gamma and beta as optim.Params.
func (l *LayerNorm) Params() []optim.Param {
	return []optim.Param{l.gamma, l.beta}
}
