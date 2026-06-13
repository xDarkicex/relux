package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// MLP is the two-layer feedforward block from the standard
// transformer, with the GELU activation:
//
//	h = gelu(x @ W1 + b1)
//	y = h @ W2 + b2
//
// where W1 is [dModel, dFF], W2 is [dFF, dModel]. dFF is
// typically 4x dModel (e.g. dModel=4096, dFF=16384 in
// LLaMA-7B). The GELU activation is the exact form used by
// LLaMA / Mistral / MiniMax:
//
//	gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//	gelu'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * phi(x)
type MLP struct {
	BaseModule
	dModel int
	dFF    int

	W1 optim.Param
	b1 optim.Param
	W2 optim.Param
	b2 optim.Param

	// Forward cache (Train mode). Cleared on each Forward.
	lastX    []float32 // [batch*seq, dModel]
	lastHpre []float32 // [batch*seq, dFF]  pre-activation
	lastH    []float32 // [batch*seq, dFF]  post-activation
}

// NewMLP constructs a two-layer feedforward block. Weights
// use He init (stddev = 1/sqrt(in_dim)); biases init to zero.
func NewMLP(dModel, dFF int) *MLP {
	if dModel <= 0 || dFF <= 0 {
		panic(fmt.Sprintf("NewMLP: dModel=%d, dFF=%d, both must be > 0", dModel, dFF))
	}
	initLinearBF16 := func(rows, cols int) []uint16 {
		stddev := float32(1.0 / math.Sqrt(float64(rows)))
		f32 := alloc.Float32(rows * cols)
		bf16 := alloc.Uint16(rows * cols)
		for i := range f32 {
			r := float32(((i*1103515245 + 12345) & 0x7fffffff) % 1000) / 500.0 - 1.0
			f32[i] = r * stddev
			bf16[i] = BF16FromF32(f32[i])
		}
		return bf16
	}
	mkParam := func(name string, data []uint16) optim.Param {
		return optim.Param{
			Name: name,
			Data: data,
			Grad: alloc.Float32(len(data)),
		}
	}
	zeros := func(n int) []uint16 { return alloc.Uint16(n) }
	return &MLP{
		dModel: dModel,
		dFF:    dFF,
		W1:     mkParam("mlp.W1", initLinearBF16(dModel, dFF)),
		b1:     mkParam("mlp.b1", zeros(dFF)),
		W2:     mkParam("mlp.W2", initLinearBF16(dFF, dModel)),
		b2:     mkParam("mlp.b2", zeros(dModel)),
	}
}

// Forward computes the MLP output.
func (m *MLP) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 || x.shape[2] != m.dModel {
		panic(fmt.Sprintf("MLP.Forward: shape %v, want [batch, seq, %d]", x.shape, m.dModel))
	}
	batch := x.shape[0]
	seq := x.shape[1]
	n := batch * seq

	xData, _ := x.ToF32()

	// h_pre = x @ W1 + b1, shape [n, dFF].
	hPre := alloc.Float32(n * m.dFF)
	matmulBatched3D(hPre, xData, m.W1.Data, batch, seq, m.dModel, m.dFF)
	b1Data := make([]float32, m.dFF)
	for i, b := range m.b1.Data {
		b1Data[i] = F32FromBF16(b)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m.dFF; j++ {
			hPre[i*m.dFF+j] += b1Data[j]
		}
	}
	// h = gelu(h_pre), shape [n, dFF].
	h := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		h[i] = geluExact(hPre[i])
	}

	// y = h @ W2 + b2, shape [n, dModel].
	y := ZerosF32(batch, seq, m.dModel)
	matmulBatched3D(y.DataF32(), h, m.W2.Data, batch, seq, m.dFF, m.dModel)
	b2Data := make([]float32, m.dModel)
	for i, b := range m.b2.Data {
		b2Data[i] = F32FromBF16(b)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m.dModel; j++ {
			y.DataF32()[i*m.dModel+j] += b2Data[j]
		}
	}

	if m.Mode() == Train {
		// Cache for Backward.
		m.lastX = xData
		m.lastHpre = hPre
		m.lastH = h
	} else {
		alloc.Free(xData)
		alloc.Free(hPre)
		alloc.Free(h)
	}

	return y
}

// Backward computes the input gradient and accumulates
// into W1.Grad, b1.Grad, W2.Grad, b2.Grad. gradOut has
// shape [batch, seq, dModel] (matching Forward's output).
//
// Standard MLP backward chain:
//
//	dL/dh_pre = (gradOut @ W2^T) * gelu'(h_pre)   [shape n, dFF]
//	dL/dW2 += h^T @ gradOut                        [shape dFF, dModel]
//	dL/db2 += sum_i gradOut[i]
//	dL/dW1 += x^T @ dL/dh_pre                      [shape dModel, dFF]
//	dL/db1 += sum_i dL/dh_pre[i]
//	dL/dx   = dL/dh_pre @ W1^T                     [shape n, dModel]
func (m *MLP) Backward(gradOut *Tensor) *Tensor {
	if m.lastX == nil {
		panic("MLP.Backward: Forward must be called first (Mode is not Train?)")
	}
	if gradOut.Rank() != 3 || gradOut.shape[2] != m.dModel {
		panic(fmt.Sprintf("MLP.Backward: gradOut shape %v, want [batch, seq, %d]", gradOut.shape, m.dModel))
	}
	batch := gradOut.shape[0]
	seq := gradOut.shape[1]
	n := batch * seq

	gOut, _ := gradOut.ToF32()
	w2Data := make([]float32, len(m.W2.Data))
	w1Data := make([]float32, len(m.W1.Data))
	for i, w := range m.W2.Data {
		w2Data[i] = F32FromBF16(w)
	}
	for i, w := range m.W1.Data {
		w1Data[i] = F32FromBF16(w)
	}

	// dL/dh_pre = (gradOut @ W2^T) * gelu'(h_pre).
	// W2 is [dFF, dModel]. W2^T is [dModel, dFF].
	// (gradOut @ W2^T)[i, j] = sum_k gradOut[i, k] * W2[j, k]
	dHpre := alloc.Float32(n * m.dFF)
	for i := 0; i < n; i++ {
		for j := 0; j < m.dFF; j++ {
			var sum float32
			for k := 0; k < m.dModel; k++ {
				sum += gOut[i*m.dModel+k] * w2Data[j*m.dModel+k]
			}
			dHpre[i*m.dFF+j] = sum * geluExactDeriv(m.lastHpre[i*m.dFF+j])
		}
	}

	// dL/dW2 += h^T @ gradOut. W2 is [dFF, dModel].
	// W2.Grad[j, k] += sum_i h[i, j] * gradOut[i, k]
	for j := 0; j < m.dFF; j++ {
		for k := 0; k < m.dModel; k++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastH[i*m.dFF+j] * gOut[i*m.dModel+k]
			}
			m.W2.Grad[j*m.dModel+k] += sum
		}
	}

	// dL/db2 += sum_i gradOut[i, k]
	for k := 0; k < m.dModel; k++ {
		var sum float32
		for i := 0; i < n; i++ {
			sum += gOut[i*m.dModel+k]
		}
		m.b2.Grad[k] += sum
	}

	// dL/dW1 += x^T @ dHpre. W1 is [dModel, dFF].
	// W1.Grad[r, j] += sum_i x[i, r] * dHpre[i, j]
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.dFF; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastX[i*m.dModel+r] * dHpre[i*m.dFF+j]
			}
			m.W1.Grad[r*m.dFF+j] += sum
		}
	}

	// dL/db1 += sum_i dHpre[i, j]
	for j := 0; j < m.dFF; j++ {
		var sum float32
		for i := 0; i < n; i++ {
			sum += dHpre[i*m.dFF+j]
		}
		m.b1.Grad[j] += sum
	}

	// dL/dx = dHpre @ W1^T. W1^T is [dFF, dModel].
	// dx[i, r] = sum_j dHpre[i, j] * W1[r, j]
	gradIn := ZerosF32(batch, seq, m.dModel)
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.dFF; j++ {
				sum += dHpre[i*m.dFF+j] * w1Data[r*m.dFF+j]
			}
			gradIn.DataF32()[i*m.dModel+r] = sum
		}
	}

	// Free the cache.
	alloc.Free(m.lastX)
	alloc.Free(m.lastHpre)
	alloc.Free(m.lastH)
	m.lastX = nil
	m.lastHpre = nil
	m.lastH = nil

	// Free dHpre.
	alloc.Free(dHpre)
	// Free the gOut float32 (allocated by ToF32 in case
	// the input was bf16/f64).
	if gradOut.dtype != Float32 {
		alloc.Free(gOut)
	}

	return gradIn
}

// Params returns W1, b1, W2, b2.
func (m *MLP) Params() []optim.Param {
	return []optim.Param{m.W1, m.b1, m.W2, m.b2}
}

// W1Param returns a pointer to the W1 weight.
func (m *MLP) W1Param() *optim.Param { return &m.W1 }

// B1Param returns a pointer to the b1 bias.
func (m *MLP) B1Param() *optim.Param { return &m.b1 }

// W2Param returns a pointer to the W2 weight.
func (m *MLP) W2Param() *optim.Param { return &m.W2 }

// B2Param returns a pointer to the b2 bias.
func (m *MLP) B2Param() *optim.Param { return &m.b2 }

// geluExact computes the exact GELU: 0.5 * x * (1 + erf(x/sqrt(2))).
func geluExact(x float32) float32 {
	return 0.5 * x * (1.0 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

// geluExactDeriv computes the derivative of the exact GELU.
// gelu'(x) = 0.5*(1 + erf(x/sqrt(2))) + x * phi(x)
// where phi(x) = exp(-x^2/2) / sqrt(2*pi).
func geluExactDeriv(x float32) float32 {
	cdf := 0.5 * (1.0 + float32(math.Erf(float64(x)/math.Sqrt2)))
	pdf := float32(math.Exp(-0.5*float64(x)*float64(x)) / math.Sqrt(2*math.Pi))
	return cdf + x*pdf
}
