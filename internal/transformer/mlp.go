package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// FFNType selects the feedforward variant.
type FFNType int

const (
	FFNGELU   FFNType = 0 // standard GELU (default)
	FFNSwiGLU FFNType = 1 // SwiGLU: (SiLU(xW_up) ⊙ xW_gate) W_down
)

// MLP is the feedforward block supporting both GELU and SwiGLU:
//
//	GELU:   h = gelu(x @ W1 + b1)
//	        y = h @ W2 + b2
//	SwiGLU: gate = SiLU(x @ W1 + b1)     // W1 used as up-projection
//	        h = gate .* (x @ WGate)      // element-wise
//	        y = h @ W2 + b2              // W2 used as down-projection
//
// SwiGLU is the standard FFN in LLaMA 2/3, Mistral, DeepSeek,
// Qwen2, and Phi-3. dFF for SwiGLU is typically 8/3 * dModel.
type MLP struct {
	BaseModule
	dModel  int
	dFF     int
	ffnType FFNType

	W1   optim.Param // GELU: up [dModel,dFF]; SwiGLU: up [dModel,dFF]
	b1   optim.Param // biases (nil for SwiGLU)
	W2   optim.Param // down [dFF,dModel]
	b2   optim.Param // biases (nil for SwiGLU)
	WGate optim.Param // SwiGLU gate [dModel,dFF] (nil for GELU)

	// Forward cache (Train mode).
	lastX    []float32 // [batch*seq, dModel]
	lastHpre []float32 // [batch*seq, dFF] pre-act (GELU) or up (SwiGLU)
	lastH    []float32 // [batch*seq, dFF] post-act (GELU) or gate*gated (SwiGLU)
	lastGate []float32 // SwiGLU only: SiLU(up) [batch*seq, dFF]
}

// NewMLP constructs a feedforward block. ffnType selects GELU
// or SwiGLU. dFF is the intermediate dimension; for SwiGLU
// typical values are 8/3 * dModel rounded to a multiple of 256.
func NewMLP(dModel, dFF int, ffnType FFNType) *MLP {
	if dModel <= 0 || dFF <= 0 {
		panic(fmt.Sprintf("NewMLP: dModel=%d, dFF=%d", dModel, dFF))
	}
	initLinearBF16 := func(rows, cols int) []uint16 {
		stddev := float32(1.0 / math.Sqrt(float64(rows)))
		f32 := alloc.Float32(rows * cols)
		bf16 := alloc.Uint16(rows * cols)
		for i := range f32 {
			r := float32(((i*1103515245+12345)&0x7fffffff)%1000)/500.0 - 1.0
			f32[i] = r * stddev
			bf16[i] = BF16FromF32(f32[i])
		}
		alloc.Free(f32)
		return bf16
	}
	mkParam := func(name string, data []uint16) optim.Param {
		return optim.Param{Name: name, Data: data, Grad: alloc.Float32(len(data))}
	}
	zeros := func(n int) []uint16 { return alloc.Uint16(n) }
	m := &MLP{
		dModel:  dModel,
		dFF:     dFF,
		ffnType: ffnType,
		W1:      mkParam("mlp.W1", initLinearBF16(dModel, dFF)),
		b1:      mkParam("mlp.b1", zeros(dFF)),
		W2:      mkParam("mlp.W2", initLinearBF16(dFF, dModel)),
		b2:      mkParam("mlp.b2", zeros(dModel)),
	}
	if ffnType == FFNSwiGLU {
		m.WGate = mkParam("mlp.W_gate", initLinearBF16(dModel, dFF))
	}
	return m
}

// Forward computes the MLP output (GELU or SwiGLU).
func (m *MLP) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 || x.shape[2] != m.dModel {
		panic(fmt.Sprintf("MLP.Forward: shape %v, want [batch, seq, %d]", x.shape, m.dModel))
	}
	batch := x.shape[0]
	seq := x.shape[1]
	n := batch * seq

	xData, _ := x.ToF32()

	if m.ffnType == FFNSwiGLU {
		return m.forwardSwiGLU(xData, batch, seq, n)
	}
	return m.forwardGELU(xData, batch, seq, n)
}

func (m *MLP) forwardGELU(xData []float32, batch, seq, n int) *Tensor {
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
	h := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		h[i] = geluExact(hPre[i])
	}
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

func (m *MLP) forwardSwiGLU(xData []float32, batch, seq, n int) *Tensor {
	// up = x @ W1  [n, dFF]
	up := alloc.Float32(n * m.dFF)
	matmulBatched3D(up, xData, m.W1.Data, batch, seq, m.dModel, m.dFF)
	// gate = SiLU(up)  [n, dFF]
	gate := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		gate[i] = silu(up[i])
	}
	// gated = x @ WGate  [n, dFF]
	gated := alloc.Float32(n * m.dFF)
	matmulBatched3D(gated, xData, m.WGate.Data, batch, seq, m.dModel, m.dFF)
	// h = gate .* gated
	h := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		h[i] = gate[i] * gated[i]
	}
	// y = h @ W2  [n, dModel]
	y := ZerosF32(batch, seq, m.dModel)
	matmulBatched3D(y.DataF32(), h, m.W2.Data, batch, seq, m.dFF, m.dModel)
	if m.Mode() == Train {
		m.lastX = xData
		m.lastHpre = up     // pre-activation (up)
		m.lastH = h         // gated output
		m.lastGate = gate   // SiLU(up)
	} else {
		alloc.Free(xData)
		alloc.Free(up)
		alloc.Free(gate)
		alloc.Free(gated)
		alloc.Free(h)
	}
	alloc.Free(gated) // always free gated (not needed for backward)
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
	var gradIn *Tensor
	if m.ffnType == FFNSwiGLU {
		gradIn = m.backwardSwiGLU(gradOut)
	} else {
		gradIn = m.backwardGELU(gradOut)
	}
	// Free the cache (inline, matching MHA.Backward pattern).
	alloc.Free(m.lastX)
	alloc.Free(m.lastHpre)
	alloc.Free(m.lastH)
	m.lastX = nil
	m.lastHpre = nil
	m.lastH = nil
	if m.lastGate != nil {
		alloc.Free(m.lastGate)
		m.lastGate = nil
	}
	return gradIn
}

func (m *MLP) backwardGELU(gradOut *Tensor) *Tensor {
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

	alloc.Free(dHpre)
	if gradOut.dtype != Float32 {
		alloc.Free(gOut)
	}
	return gradIn
}

// backwardSwiGLU computes gradients for the SwiGLU path.
func (m *MLP) backwardSwiGLU(gradOut *Tensor) *Tensor {
	batch := gradOut.shape[0]
	seq := gradOut.shape[1]
	n := batch * seq

	gOut, _ := gradOut.ToF32()
	w2F32 := make([]float32, len(m.W2.Data))
	w1F32 := make([]float32, len(m.W1.Data))
	wGateF32 := make([]float32, len(m.WGate.Data))
	for i, w := range m.W2.Data {
		w2F32[i] = F32FromBF16(w)
	}
	for i, w := range m.W1.Data {
		w1F32[i] = F32FromBF16(w)
	}
	for i, w := range m.WGate.Data {
		wGateF32[i] = F32FromBF16(w)
	}

	// dL/dH = gradOut @ WDown^T  [n, dFF]
	dH := alloc.Float32(n * m.dFF)
	for i := 0; i < n; i++ {
		for j := 0; j < m.dFF; j++ {
			var sum float32
			for k := 0; k < m.dModel; k++ {
				sum += gOut[i*m.dModel+k] * w2F32[j*m.dModel+k]
			}
			dH[i*m.dFF+j] = sum
		}
	}

	// dL/dW2 += H^T @ gradOut
	for j := 0; j < m.dFF; j++ {
		for k := 0; k < m.dModel; k++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastH[i*m.dFF+j] * gOut[i*m.dModel+k]
			}
			m.W2.Grad[j*m.dModel+k] += sum
		}
	}

	// gated = h / gate  (since h = gate * gated)
	// dL/d_gate = dH * gated, dL/d_gated = dH * gate
	eps := float32(1e-12)
	dGate := alloc.Float32(n * m.dFF)
	dGated := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		gatedI := m.lastH[i] / (m.lastGate[i] + eps)
		dGate[i] = dH[i] * gatedI
		dGated[i] = dH[i] * m.lastGate[i]
	}

	// dL/d_up = dGate * SiLU'(up)
	dUp := alloc.Float32(n * m.dFF)
	for i := 0; i < n*m.dFF; i++ {
		dUp[i] = dGate[i] * siluDeriv(m.lastHpre[i])
	}

	// dL/d_WUp (W1) += x^T @ dUp
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.dFF; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastX[i*m.dModel+r] * dUp[i*m.dFF+j]
			}
			m.W1.Grad[r*m.dFF+j] += sum
		}
	}

	// dL/d_WGate += x^T @ dGated
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.dFF; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastX[i*m.dModel+r] * dGated[i*m.dFF+j]
			}
			m.WGate.Grad[r*m.dFF+j] += sum
		}
	}

	// dL/d_x = dUp @ WUp^T + dGated @ WGate^T
	gradIn := ZerosF32(batch, seq, m.dModel)
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.dFF; j++ {
				sum += dUp[i*m.dFF+j]*w1F32[r*m.dFF+j] + dGated[i*m.dFF+j]*wGateF32[r*m.dFF+j]
			}
			gradIn.DataF32()[i*m.dModel+r] = sum
		}
	}

	alloc.Free(dH)
	alloc.Free(dGate)
	alloc.Free(dGated)
	alloc.Free(dUp)
	if gradOut.dtype != Float32 {
		alloc.Free(gOut)
	}
	return gradIn
}

// freeForwardCache releases the activation cache.
func (m *MLP) freeForwardCache() {
	if m.lastX != nil {
		alloc.Free(m.lastX)
		alloc.Free(m.lastHpre)
		alloc.Free(m.lastH)
		m.lastX = nil
		m.lastHpre = nil
		m.lastH = nil
	}
	if m.lastGate != nil {
		alloc.Free(m.lastGate)
		m.lastGate = nil
	}
}

func (m *MLP) Params() []optim.Param {
	if m.ffnType == FFNSwiGLU {
		return []optim.Param{m.W1, m.W2, m.WGate}
	}
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

// silu computes SiLU(x) = x * sigmoid(x).
func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

// siluDeriv computes dSiLU/dx = sigmoid(x) * (1 + x*(1-sig(x))).
func siluDeriv(x float32) float32 {
	sig := float32(1.0 / (1.0 + float32(math.Exp(float64(-x)))))
	return sig * (1.0 + x*(1.0-sig))
}
