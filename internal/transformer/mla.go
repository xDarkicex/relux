package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// MLA is Multi-head Latent Attention. It compresses the KV cache by
// projecting the hidden state into a low-rank latent c^KV and storing
// only that latent + a small decoupled RoPE key per position. Full K
// and V are reconstructed on-the-fly during attention.
//
// Decoupled RoPE: content subkeys (K^C, Q^C) carry semantic information
// and are NOT rotated. Position subkeys (K^R, Q^R) are small per-head
// vectors that carry relative position via RoPE. The attention score is:
//
//	(q_i^C · k_j^C + q_i^R · k_j^R) / √(headDim + dHR)
//
// K^R is one dHR-dim vector shared across all heads. Q_i^R is per-head.
// V has no RoPE component (position doesn't matter for value mixing).
type MLA struct {
	BaseModule
	dModel   int
	numHeads int
	headDim  int
	dC       int // KV compression dimension
	dHR      int // RoPE dimension per head
	causal   bool

	// ropeKR applies decoupled RoPE to Q^R and K^R (headDim = dHR).
	ropeKR *RotaryEmbedding

	FlashAttention bool

	// Weight matrices (bf16 Data, f32 Grad).
	W_Q   optim.Param // [dModel, numHeads*headDim] query content
	W_DKV optim.Param // [dModel, dC] joint KV down-projection
	W_UK  optim.Param // [dC, numHeads*headDim] key up-projection
	W_UV  optim.Param // [dC, numHeads*headDim] value up-projection
	W_KR  optim.Param // [dModel, dHR] RoPE key (shared across heads)
	W_QR  optim.Param // [dModel, numHeads*dHR] RoPE query (per-head)
	W_O   optim.Param // [numHeads*headDim, dModel] output projection

	// Inference-only compressed KV cache.
	Cache    *MLACache
	LayerIdx int

	// Forward cache (Train mode).
	lastX           []float32 // [batch*seq, dModel] input
	lastQc          []float32 // [batch, numHeads, seq, headDim] query content
	lastQr          []float32 // [batch, numHeads, seq, dHR] query RoPE (post-RoPE)
	lastKc          []float32 // [batch, numHeads, seq, headDim] key content
	lastKr          []float32 // [batch, numHeads, seq, dHR] key RoPE (post-RoPE, broadcast)
	lastV           []float32 // [batch, numHeads, seq, headDim] value
	lastRearranged  []float32 // [batch*seq, numHeads*headDim] pre-W_O
	lastAttn        []float32 // [batch, numHeads, seq, seq] softmax(weights)
	lastInSeqLen    int
}

// NewMLA constructs a Multi-head Latent Attention module.
// dC is the KV compression dimension (typically 4×headDim).
// dHR is the decoupled RoPE dimension (typically headDim/2, must be even).
// maxSeqLen and ropeBase are for the internal dHR RoPE module.
func NewMLA(dModel, numHeads, dC, dHR int, maxSeqLen int, ropeBase float32, causal bool) *MLA {
	if dModel <= 0 || numHeads <= 0 || dC <= 0 || dHR <= 0 {
		panic(fmt.Sprintf("NewMLA: dModel=%d numHeads=%d dC=%d dHR=%d, all must be > 0", dModel, numHeads, dC, dHR))
	}
	if dModel%numHeads != 0 {
		panic(fmt.Sprintf("NewMLA: dModel=%d not divisible by numHeads=%d", dModel, numHeads))
	}
	if dHR%2 != 0 {
		panic(fmt.Sprintf("NewMLA: dHR=%d must be even for paired RoPE rotation", dHR))
	}
	if ropeBase == 0 {
		ropeBase = 10000
	}

	headDim := dModel / numHeads
	ropeKR := NewRotaryEmbedding(dHR, ropeBase, maxSeqLen)

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

	return &MLA{
		dModel:   dModel,
		numHeads: numHeads,
		headDim:  headDim,
		dC:       dC,
		dHR:      dHR,
		causal:   causal,
		ropeKR:   ropeKR,
		W_Q:      mkParam("mla.W_Q", initLinearBF16(dModel, numHeads*headDim)),
		W_DKV:    mkParam("mla.W_DKV", initLinearBF16(dModel, dC)),
		W_UK:     mkParam("mla.W_UK", initLinearBF16(dC, numHeads*headDim)),
		W_UV:     mkParam("mla.W_UV", initLinearBF16(dC, numHeads*headDim)),
		W_KR:     mkParam("mla.W_KR", initLinearBF16(dModel, dHR)),
		W_QR:     mkParam("mla.W_QR", initLinearBF16(dModel, numHeads*dHR)),
		W_O:      mkParam("mla.W_O", initLinearBF16(numHeads*headDim, dModel)),
	}
}

// Forward computes the MLA output. Input shape [batch, seq, dModel] →
// output shape [batch, seq, dModel].
func (m *MLA) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 {
		panic(fmt.Sprintf("MLA.Forward: input rank=%d, want 3", x.Rank()))
	}
	if x.shape[2] != m.dModel {
		panic(fmt.Sprintf("MLA.Forward: input last dim=%d, want %d", x.shape[2], m.dModel))
	}
	if m.Mode() == Inference && m.Cache != nil {
		return m.forwardCached(x)
	}

	batch := x.shape[0]
	seq := x.shape[1]
	n := batch * seq

	xData, _ := x.ToF32()

	// 1. Linear projections.
	qcFlat := alloc.Float32(n * m.numHeads * m.headDim)
	ckvFlat := alloc.Float32(n * m.dC)
	qrFlat := alloc.Float32(n * m.numHeads * m.dHR)
	krFlat := alloc.Float32(n * m.dHR)

	matmulBatched3D(qcFlat, xData, m.W_Q.Data, batch, seq, m.dModel, m.numHeads*m.headDim)
	matmulBatched3D(ckvFlat, xData, m.W_DKV.Data, batch, seq, m.dModel, m.dC)
	matmulBatched3D(qrFlat, xData, m.W_QR.Data, batch, seq, m.dModel, m.numHeads*m.dHR)
	matmulBatched3D(krFlat, xData, m.W_KR.Data, batch, seq, m.dModel, m.dHR)

	// 2. Decompress K, V from latent.
	kcFlat := alloc.Float32(n * m.numHeads * m.headDim)
	vFlat := alloc.Float32(n * m.numHeads * m.headDim)
	matmulBatched3D(kcFlat, ckvFlat, m.W_UK.Data, batch, seq, m.dC, m.numHeads*m.headDim)
	matmulBatched3D(vFlat, ckvFlat, m.W_UV.Data, batch, seq, m.dC, m.numHeads*m.headDim)
	alloc.Free(ckvFlat)

	// 3. Rearrange to [batch, numHeads, seq, dim].
	qc := rearrangeBSNH(qcFlat, batch, seq, m.numHeads, m.headDim)
	kc := rearrangeBSNH(kcFlat, batch, seq, m.numHeads, m.headDim)
	v := rearrangeBSNH(vFlat, batch, seq, m.numHeads, m.headDim)
	qr := rearrangeBSNH(qrFlat, batch, seq, m.numHeads, m.dHR)

	alloc.Free(qcFlat)
	alloc.Free(kcFlat)
	alloc.Free(vFlat)
	alloc.Free(qrFlat)

	// 4. Apply RoPE to Q^R and K^R.
	// kr is [batch*seq, dHR] — reshape to [batch, 1, seq, dHR] for RoPE.
	kr4D := alloc.Float32(batch * 1 * seq * m.dHR)
	copy(kr4D, krFlat)
	alloc.Free(krFlat)

	qrT := &Tensor{shape: []int{batch, m.numHeads, seq, m.dHR}, dtype: Float32, f32: qr}
	qrRopeT := m.ropeKR.Apply(qrT, 0)
	copy(qr, qrRopeT.f32)
	alloc.Free(qrRopeT.f32)

	krT := &Tensor{shape: []int{batch, 1, seq, m.dHR}, dtype: Float32, f32: kr4D}
	krRopeT := m.ropeKR.Apply(krT, 0)
	copy(kr4D, krRopeT.f32)
	alloc.Free(krRopeT.f32)

	// 5. Broadcast K^R from [batch, 1, seq, dHR] to [batch, numHeads, seq, dHR].
	krBroadcast := alloc.Float32(batch * m.numHeads * seq * m.dHR)
	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			src := kr4D[(b*1+0)*seq*m.dHR : (b*1+0)*seq*m.dHR+seq*m.dHR]
			dst := krBroadcast[(b*m.numHeads+h)*seq*m.dHR : (b*m.numHeads+h)*seq*m.dHR+seq*m.dHR]
			copy(dst, src)
		}
	}
	alloc.Free(kr4D)

	// 6. Attention with scale 1/√(headDim + dHR).
	// Per head: S = (Qc @ Kc^T) + (Qr @ Kr^T), then scale, causal mask, softmax.
	attnDim := m.headDim + m.dHR
	scale := float32(1.0 / math.Sqrt(float64(attnDim)))

	attnOut := alloc.Float32(batch * m.numHeads * seq * m.headDim)
	attnWeights := alloc.Float32(batch * m.numHeads * seq * seq)

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			qcOff := (b*m.numHeads + h) * seq * m.headDim
			kcOff := (b*m.numHeads + h) * seq * m.headDim
			qrOff := (b*m.numHeads + h) * seq * m.dHR
			krOff := (b*m.numHeads + h) * seq * m.dHR
			vOff := (b*m.numHeads + h) * seq * m.headDim
			wOff := (b*m.numHeads + h) * seq * seq
			oOff := (b*m.numHeads + h) * seq * m.headDim

			// S = Qc @ Kc^T.
			matmulFloat32TransB(attnWeights[wOff:wOff+seq*seq], qc[qcOff:qcOff+seq*m.headDim], kc[kcOff:kcOff+seq*m.headDim], seq, m.headDim, seq)

			// S += Qr @ Kr^T.
			scoresQrKr := alloc.Float32(seq * seq)
			matmulFloat32TransB(scoresQrKr, qr[qrOff:qrOff+seq*m.dHR], krBroadcast[krOff:krOff+seq*m.dHR], seq, m.dHR, seq)
			for i := 0; i < seq*seq; i++ {
				attnWeights[wOff+i] += scoresQrKr[i]
			}
			alloc.Free(scoresQrKr)

			// Scale and causal mask.
			for i := 0; i < seq; i++ {
				for j := 0; j < seq; j++ {
					attnWeights[wOff+i*seq+j] *= scale
					if m.causal && j > i {
						attnWeights[wOff+i*seq+j] = float32(math.Inf(-1))
					}
				}
			}

			softmaxRows(attnWeights[wOff:wOff+seq*seq], seq, seq)
			matmulFloat32(attnOut[oOff:oOff+seq*m.headDim], attnWeights[wOff:wOff+seq*seq], v[vOff:vOff+seq*m.headDim], seq, seq, m.headDim)
		}
	}

	// 7. Rearrange output back + W_O projection.
	rearrangedBack := rearrangeBHSN(attnOut, batch, m.numHeads, seq, m.headDim)
	alloc.Free(attnOut)

	out := ZerosF32(batch, seq, m.dModel)
	matmulBatched3D(out.DataF32(), rearrangedBack, m.W_O.Data, batch, seq, m.numHeads*m.headDim, m.dModel)

	if m.Mode() == Train {
		m.lastX = xData
		m.lastQc = qc
		m.lastQr = qr
		m.lastKc = kc
		m.lastKr = krBroadcast
		m.lastV = v
		m.lastRearranged = rearrangedBack
		m.lastAttn = attnWeights
		m.lastInSeqLen = seq
	} else {
		alloc.Free(xData)
		alloc.Free(qc)
		alloc.Free(qr)
		alloc.Free(kc)
		alloc.Free(krBroadcast)
		alloc.Free(v)
		alloc.Free(rearrangedBack)
		alloc.Free(attnWeights)
	}

	return out
}

// forwardCached is the inference-only forward with compressed KV cache.
func (m *MLA) forwardCached(x *Tensor) *Tensor {
	batch := x.shape[0]
	seqQ := x.shape[1]
	n := batch * seqQ

	xData, _ := x.ToF32()

	// 1. Project new token(s).
	qcFlat := alloc.Float32(n * m.numHeads * m.headDim)
	qrFlat := alloc.Float32(n * m.numHeads * m.dHR)
	ckvFlat := alloc.Float32(n * m.dC)
	krFlat := alloc.Float32(n * m.dHR)

	matmulBatched3D(qcFlat, xData, m.W_Q.Data, batch, seqQ, m.dModel, m.numHeads*m.headDim)
	matmulBatched3D(qrFlat, xData, m.W_QR.Data, batch, seqQ, m.dModel, m.numHeads*m.dHR)
	matmulBatched3D(ckvFlat, xData, m.W_DKV.Data, batch, seqQ, m.dModel, m.dC)
	matmulBatched3D(krFlat, xData, m.W_KR.Data, batch, seqQ, m.dModel, m.dHR)

	// 2. Rearrange Q to [batch, numHeads, seqQ, dim].
	qc := rearrangeBSNH(qcFlat, batch, seqQ, m.numHeads, m.headDim)
	qr := rearrangeBSNH(qrFlat, batch, seqQ, m.numHeads, m.dHR)
	alloc.Free(qcFlat)
	alloc.Free(qrFlat)

	// 3. RoPE on Q^R and K^R with offset = cacheLen.
	cacheLen := m.Cache.TotalLen(m.LayerIdx)

	qrT := &Tensor{shape: []int{batch, m.numHeads, seqQ, m.dHR}, dtype: Float32, f32: qr}
	qrRopeT := m.ropeKR.Apply(qrT, cacheLen)
	copy(qr, qrRopeT.f32)
	alloc.Free(qrRopeT.f32)

	kr1D := &Tensor{shape: []int{n, m.dHR}, dtype: Float32, f32: krFlat}
	krRope1D := m.ropeKR.Apply(kr1D, cacheLen)
	copy(krFlat, krRope1D.f32)
	alloc.Free(krRope1D.f32)

	// 4. Append c^KV and K^R (post-RoPE) to cache.
	ckvT := &Tensor{shape: []int{batch, seqQ, m.dC}, dtype: Float32, f32: ckvFlat}
	krCacheT := &Tensor{shape: []int{batch, seqQ, m.dHR}, dtype: Float32, f32: krFlat}
	m.Cache.Append(m.LayerIdx, ckvT, krCacheT)
	alloc.Free(ckvFlat)
	alloc.Free(krFlat)

	// 5. Read full cached state → decompress.
	cachedC, cachedKR := m.Cache.View(m.LayerIdx)
	cachedCData, _ := cachedC.ToF32()
	cachedKRData, _ := cachedKR.ToF32()
	totalLen := cachedC.shape[len(cachedC.shape)-2]

	// Decompress all cached latents → K^C, V.
	kcAll := alloc.Float32(batch * m.numHeads * totalLen * m.headDim)
	vAll := alloc.Float32(batch * m.numHeads * totalLen * m.headDim)
	matmulBatched3D(kcAll, cachedCData, m.W_UK.Data, batch, totalLen, m.dC, m.numHeads*m.headDim)
	matmulBatched3D(vAll, cachedCData, m.W_UV.Data, batch, totalLen, m.dC, m.numHeads*m.headDim)

	// Broadcast K^R across heads: [batch, 1, totalLen, dHR] → [batch, numHeads, totalLen, dHR].
	krAll := alloc.Float32(batch * m.numHeads * totalLen * m.dHR)
	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			src := cachedKRData[(b*1+0)*totalLen*m.dHR : (b*1+0)*totalLen*m.dHR+totalLen*m.dHR]
			dst := krAll[(b*m.numHeads+h)*totalLen*m.dHR : (b*m.numHeads+h)*totalLen*m.dHR+totalLen*m.dHR]
			copy(dst, src)
		}
	}

	// 6. Attention with full cached sequence.
	attnDim := m.headDim + m.dHR
	scale := float32(1.0 / math.Sqrt(float64(attnDim)))

	attnOut := alloc.Float32(batch * m.numHeads * seqQ * m.headDim)
	attnWeights := alloc.Float32(batch * m.numHeads * seqQ * totalLen)

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			qcOff := (b*m.numHeads + h) * seqQ * m.headDim
			kcOff := (b*m.numHeads + h) * totalLen * m.headDim
			qrOff := (b*m.numHeads + h) * seqQ * m.dHR
			krOff := (b*m.numHeads + h) * totalLen * m.dHR
			vOff := (b*m.numHeads + h) * totalLen * m.headDim
			wOff := (b*m.numHeads + h) * seqQ * totalLen
			oOff := (b*m.numHeads + h) * seqQ * m.headDim

			// S = Qc @ Kc^T + Qr @ Kr^T.
			matmulFloat32TransB(attnWeights[wOff:wOff+seqQ*totalLen], qc[qcOff:qcOff+seqQ*m.headDim], kcAll[kcOff:kcOff+totalLen*m.headDim], seqQ, m.headDim, totalLen)

			scoresQrKr := alloc.Float32(seqQ * totalLen)
			matmulFloat32TransB(scoresQrKr, qr[qrOff:qrOff+seqQ*m.dHR], krAll[krOff:krOff+totalLen*m.dHR], seqQ, m.dHR, totalLen)
			for i := 0; i < seqQ*totalLen; i++ {
				attnWeights[wOff+i] += scoresQrKr[i]
			}
			alloc.Free(scoresQrKr)

			for i := 0; i < seqQ; i++ {
				for j := 0; j < totalLen; j++ {
					attnWeights[wOff+i*totalLen+j] *= scale
					if m.causal && seqQ == totalLen && j > i {
						attnWeights[wOff+i*totalLen+j] = float32(math.Inf(-1))
					}
				}
			}

			softmaxRows(attnWeights[wOff:wOff+seqQ*totalLen], seqQ, totalLen)
			matmulFloat32(attnOut[oOff:oOff+seqQ*m.headDim], attnWeights[wOff:wOff+seqQ*totalLen], vAll[vOff:vOff+totalLen*m.headDim], seqQ, totalLen, m.headDim)
		}
	}

	alloc.Free(kcAll)
	alloc.Free(vAll)
	alloc.Free(krAll)
	alloc.Free(attnWeights)

	// 7. Rearrange + W_O.
	rearrangedBack := rearrangeBHSN(attnOut, batch, m.numHeads, seqQ, m.headDim)
	alloc.Free(attnOut)

	out := ZerosF32(batch, seqQ, m.dModel)
	matmulBatched3D(out.DataF32(), rearrangedBack, m.W_O.Data, batch, seqQ, m.numHeads*m.headDim, m.dModel)

	alloc.Free(xData)
	alloc.Free(qc)
	alloc.Free(qr)
	alloc.Free(rearrangedBack)

	return out
}

// Backward computes the input gradient and accumulates into all weight
// gradients. gradOut has shape [batch, seq, dModel].
func (m *MLA) Backward(gradOut *Tensor) *Tensor {
	if m.lastX == nil {
		panic("MLA.Backward: Forward must be called first (Mode is not Train?)")
	}
	if gradOut.Rank() != 3 || gradOut.shape[2] != m.dModel {
		panic(fmt.Sprintf("MLA.Backward: gradOut shape %v, want [batch, seq, %d]", gradOut.shape, m.dModel))
	}

	batch := gradOut.shape[0]
	seq := gradOut.shape[1]
	n := batch * seq

	gOut, _ := gradOut.ToF32()
	xData := m.lastX

	// Widen weights for backward matmuls.
	widen := func(data []uint16) []float32 {
		out := make([]float32, len(data))
		for i, d := range data {
			out[i] = F32FromBF16(d)
		}
		return out
	}
	wq := widen(m.W_Q.Data)
	wdkv := widen(m.W_DKV.Data)
	wuk := widen(m.W_UK.Data)
	wuv := widen(m.W_UV.Data)
	wkr := widen(m.W_KR.Data)
	wqr := widen(m.W_QR.Data)
	wo := widen(m.W_O.Data)

	// Step 1: dRearranged = gradOut @ Wo^T. Wo is [numHeads*headDim, dModel].
	dRearranged := alloc.Float32(n * m.numHeads * m.headDim)
	for i := 0; i < n; i++ {
		for j := 0; j < m.numHeads*m.headDim; j++ {
			var sum float32
			for k := 0; k < m.dModel; k++ {
				sum += gOut[i*m.dModel+k] * wo[j*m.dModel+k]
			}
			dRearranged[i*m.numHeads*m.headDim+j] = sum
		}
	}

	// Wo.Grad += sum_n rearrangedBack[n] outer gradOut[n].
	for j := 0; j < m.numHeads*m.headDim; j++ {
		for k := 0; k < m.dModel; k++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastRearranged[i*m.numHeads*m.headDim+j] * gOut[i*m.dModel+k]
			}
			m.W_O.Grad[j*m.dModel+k] += sum
		}
	}

	// Step 2: dAttnOut = rearrangeBSNH(dRearranged).
	dAttnOut := rearrangeBSNH(dRearranged, batch, m.numHeads, seq, m.headDim)
	alloc.Free(dRearranged)

	// Step 3: dAttnW and dV from attention output.
	dAttnW := alloc.Float32(batch * m.numHeads * seq * seq)
	dV := alloc.Float32(batch * m.numHeads * seq * m.headDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			dAOff := (b*m.numHeads + h) * seq * m.headDim
			dVOff := (b*m.numHeads + h) * seq * m.headDim
			dWOff := (b*m.numHeads + h) * seq * seq
			aOff := (b*m.numHeads + h) * seq * seq
			vOff := (b*m.numHeads + h) * seq * m.headDim

			// dAttnW[i,j] = sum_k dO[i,k] * V[j,k].
			for i := 0; i < seq; i++ {
				for j := 0; j < seq; j++ {
					var sum float32
					for k := 0; k < m.headDim; k++ {
						sum += dAttnOut[dAOff+i*m.headDim+k] * m.lastV[vOff+j*m.headDim+k]
					}
					dAttnW[dWOff+i*seq+j] = sum
				}
			}
			// dV[j,k] = sum_i attnW[i,j] * dO[i,k].
			for j := 0; j < seq; j++ {
				for k := 0; k < m.headDim; k++ {
					var sum float32
					for i := 0; i < seq; i++ {
						sum += m.lastAttn[aOff+i*seq+j] * dAttnOut[dAOff+i*m.headDim+k]
					}
					dV[dVOff+j*m.headDim+k] = sum
				}
			}
		}
	}
	alloc.Free(dAttnOut)

	// Step 4: Softmax backward → dScores.
	scale := float32(1.0 / math.Sqrt(float64(m.headDim+m.dHR)))
	dScores := alloc.Float32(batch * m.numHeads * seq * seq)
	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			wOff := (b*m.numHeads + h) * seq * seq
			for i := 0; i < seq; i++ {
				var rowSum float32
				for k := 0; k < seq; k++ {
					rowSum += dAttnW[wOff+i*seq+k] * m.lastAttn[wOff+i*seq+k]
				}
				for j := 0; j < seq; j++ {
					attnIJ := m.lastAttn[wOff+i*seq+j]
					dScores[wOff+i*seq+j] = attnIJ * (dAttnW[wOff+i*seq+j] - rowSum)
				}
			}
		}
	}
	alloc.Free(dAttnW)

	// Step 5: dQc, dQr, dKc, dKr from dScores.
	// dQc[b,h,i,k] = sum_j dScores[i,j] * Kc[j,k] * scale
	// dQr[b,h,i,k] = sum_j dScores[i,j] * Kr[j,k] * scale
	// dKc[b,h,j,k] = sum_i dScores[i,j] * Qc[i,k] * scale
	// dKr_per_head[b,h,j,k] = sum_i dScores[i,j] * Qr[i,k] * scale
	dQc := alloc.Float32(batch * m.numHeads * seq * m.headDim)
	dQr := alloc.Float32(batch * m.numHeads * seq * m.dHR)
	dKc := alloc.Float32(batch * m.numHeads * seq * m.headDim)
	dKrPerHead := alloc.Float32(batch * m.numHeads * seq * m.dHR)

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			qcOff := (b*m.numHeads + h) * seq * m.headDim
			kcOff := (b*m.numHeads + h) * seq * m.headDim
			qrOff := (b*m.numHeads + h) * seq * m.dHR
			krOff := (b*m.numHeads + h) * seq * m.dHR
			sOff := (b*m.numHeads + h) * seq * seq
			dQcOff := (b*m.numHeads + h) * seq * m.headDim
			dKcOff := (b*m.numHeads + h) * seq * m.headDim
			dQrOff := (b*m.numHeads + h) * seq * m.dHR
			dKrOff := (b*m.numHeads + h) * seq * m.dHR

			// dQc and dQr (sum over j).
			for i := 0; i < seq; i++ {
				for k := 0; k < m.headDim; k++ {
					var sum float32
					for j := 0; j < seq; j++ {
						sum += dScores[sOff+i*seq+j] * m.lastKc[kcOff+j*m.headDim+k]
					}
					dQc[dQcOff+i*m.headDim+k] = sum * scale
				}
			}
			for i := 0; i < seq; i++ {
				for k := 0; k < m.dHR; k++ {
					var sum float32
					for j := 0; j < seq; j++ {
						sum += dScores[sOff+i*seq+j] * m.lastKr[krOff+j*m.dHR+k]
					}
					dQr[dQrOff+i*m.dHR+k] = sum * scale
				}
			}
			// dKc and dKr_per_head (sum over i).
			for j := 0; j < seq; j++ {
				for k := 0; k < m.headDim; k++ {
					var sum float32
					for i := 0; i < seq; i++ {
						sum += dScores[sOff+i*seq+j] * m.lastQc[qcOff+i*m.headDim+k]
					}
					dKc[dKcOff+j*m.headDim+k] = sum * scale
				}
			}
			for j := 0; j < seq; j++ {
				for k := 0; k < m.dHR; k++ {
					var sum float32
					for i := 0; i < seq; i++ {
						sum += dScores[sOff+i*seq+j] * m.lastQr[qrOff+i*m.dHR+k]
					}
					dKrPerHead[dKrOff+j*m.dHR+k] = sum * scale
				}
			}
		}
	}
	alloc.Free(dScores)

	// Step 6: Sum dKr across heads (K^R is shared, broadcast in forward).
	dKr := alloc.Float32(batch * 1 * seq * m.dHR)
	for b := 0; b < batch; b++ {
		for j := 0; j < seq; j++ {
			for k := 0; k < m.dHR; k++ {
				var sum float32
				for h := 0; h < m.numHeads; h++ {
					sum += dKrPerHead[(b*m.numHeads+h)*seq*m.dHR+j*m.dHR+k]
				}
				dKr[(b*1+0)*seq*m.dHR+j*m.dHR+k] = sum
			}
		}
	}
	alloc.Free(dKrPerHead)

	// Step 7: RoPE backward on dQr and dKr.
	dQrT := &Tensor{shape: []int{batch, m.numHeads, seq, m.dHR}, dtype: Float32, f32: dQr}
	dQrRope := m.ropeKR.BackwardApply(dQrT, 0)
	copy(dQr, dQrRope.f32)
	alloc.Free(dQrRope.f32)

	// dKr is [batch, 1, seq, dHR] — reshape to match RoPE shape expectation.
	// RoPEBackwardApply with startPos=0 handles [batch, 1, seq, dHR].
	dKrT := &Tensor{shape: []int{batch, 1, seq, m.dHR}, dtype: Float32, f32: dKr}
	dKrRope := m.ropeKR.BackwardApply(dKrT, 0)
	copy(dKr, dKrRope.f32)
	alloc.Free(dKrRope.f32)

	// Step 8: Rearrange gradients back to [batch*seq, dim].
	dQcFlat := rearrangeBHSN(dQc, batch, m.numHeads, seq, m.headDim)
	dQrFlat := rearrangeBHSN(dQr, batch, m.numHeads, seq, m.dHR)
	dKcFlat := rearrangeBHSN(dKc, batch, m.numHeads, seq, m.headDim)
	dVFlat := rearrangeBHSN(dV, batch, m.numHeads, seq, m.headDim)
	// dKr is [batch, 1, seq, dHR] → flatten to [batch*seq, dHR].
	dKrFlat := alloc.Float32(n * m.dHR)
	for i := 0; i < n; i++ {
		copy(dKrFlat[i*m.dHR:(i+1)*m.dHR], dKr[i*m.dHR:(i+1)*m.dHR])
	}

	alloc.Free(dQc)
	alloc.Free(dQr)
	alloc.Free(dKc)
	alloc.Free(dV)
	alloc.Free(dKr)

	// Step 9: Gradient through decompression up-projections.
	// dL/d(c_kv) = dKcFlat @ W_UK^T + dVFlat @ W_UV^T.
	// W_UK is [dC, numHeads*headDim]. W_UK^T is [numHeads*headDim, dC].
	dCKV := alloc.Float32(n * m.dC)

	// dKcFlat @ W_UK^T contribution.
	for i := 0; i < n; i++ {
		for j := 0; j < m.dC; j++ {
			var sum float32
			for k := 0; k < m.numHeads*m.headDim; k++ {
				sum += dKcFlat[i*m.numHeads*m.headDim+k] * wuk[j*m.numHeads*m.headDim+k]
			}
			dCKV[i*m.dC+j] += sum
		}
	}
	// dVFlat @ W_UV^T contribution.
	for i := 0; i < n; i++ {
		for j := 0; j < m.dC; j++ {
			var sum float32
			for k := 0; k < m.numHeads*m.headDim; k++ {
				sum += dVFlat[i*m.numHeads*m.headDim+k] * wuv[j*m.numHeads*m.headDim+k]
			}
			dCKV[i*m.dC+j] += sum
		}
	}

	// W_UK.Grad += c_kv^T @ dKcFlat. c_kv was not stored; recompute from x.
	ckvRecomp := alloc.Float32(n * m.dC)
	matmulBatched3D(ckvRecomp, xData, m.W_DKV.Data, batch, seq, m.dModel, m.dC)

	for r := 0; r < m.dC; r++ {
		for j := 0; j < m.numHeads*m.headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += ckvRecomp[i*m.dC+r] * dKcFlat[i*m.numHeads*m.headDim+j]
			}
			m.W_UK.Grad[r*m.numHeads*m.headDim+j] += sum
		}
	}
	// W_UV.Grad += c_kv^T @ dVFlat.
	for r := 0; r < m.dC; r++ {
		for j := 0; j < m.numHeads*m.headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += ckvRecomp[i*m.dC+r] * dVFlat[i*m.numHeads*m.headDim+j]
			}
			m.W_UV.Grad[r*m.numHeads*m.headDim+j] += sum
		}
	}
	alloc.Free(ckvRecomp)

	// Step 10: Gradient through down-projection W_DKV.
	// dL/dW_DKV = x^T @ dCKV.
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.dC; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dCKV[i*m.dC+j]
			}
			m.W_DKV.Grad[r*m.dC+j] += sum
		}
	}
	alloc.Free(dCKV)

	// W_Q.Grad += x^T @ dQcFlat.
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.numHeads*m.headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dQcFlat[i*m.numHeads*m.headDim+j]
			}
			m.W_Q.Grad[r*m.numHeads*m.headDim+j] += sum
		}
	}
	// W_QR.Grad += x^T @ dQrFlat.
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.numHeads*m.dHR; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dQrFlat[i*m.numHeads*m.dHR+j]
			}
			m.W_QR.Grad[r*m.numHeads*m.dHR+j] += sum
		}
	}
	// W_KR.Grad += x^T @ dKrFlat.
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.dHR; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dKrFlat[i*m.dHR+j]
			}
			m.W_KR.Grad[r*m.dHR+j] += sum
		}
	}

	// Step 11: dL/dX = sum of all upstream paths through the weight matrices.
	gradIn := ZerosF32(batch, seq, m.dModel)
	// dQcFlat @ W_Q^T
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.numHeads*m.headDim; j++ {
				sum += dQcFlat[i*m.numHeads*m.headDim+j] * wq[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}
	// dQrFlat @ W_QR^T
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.numHeads*m.dHR; j++ {
				sum += dQrFlat[i*m.numHeads*m.dHR+j] * wqr[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}
	// dKcFlat @ W_UK^T (gradient through the K content path: x→ckv→kc)
	// Actually, dKcFlat flows back to ckv, which flows back to x through W_DKV.
	// Wait — I already computed dCKV above (the gradient to the latent).
	// dCKV @ W_DKV^T gives the x-gradient from the KV path.
	// But I freed dCKV! I need to recompute it from dKcFlat and dVFlat.

	// Recompute dCKV for the input gradient path.
	dCKV2 := alloc.Float32(n * m.dC)
	for i := 0; i < n; i++ {
		for j := 0; j < m.dC; j++ {
			var sum float32
			for k := 0; k < m.numHeads*m.headDim; k++ {
				sum += dKcFlat[i*m.numHeads*m.headDim+k] * wuk[j*m.numHeads*m.headDim+k]
			}
			dCKV2[i*m.dC+j] += sum
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m.dC; j++ {
			var sum float32
			for k := 0; k < m.numHeads*m.headDim; k++ {
				sum += dVFlat[i*m.numHeads*m.headDim+k] * wuv[j*m.numHeads*m.headDim+k]
			}
			dCKV2[i*m.dC+j] += sum
		}
	}

	// dCKV2 @ W_DKV^T → x-grad from KV path.
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.dC; j++ {
				sum += dCKV2[i*m.dC+j] * wdkv[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}
	alloc.Free(dCKV2)

	// dKrFlat @ W_KR^T → x-grad from RoPE key path.
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.dHR; j++ {
				sum += dKrFlat[i*m.dHR+j] * wkr[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}

	// Free temporaries.
	alloc.Free(dQcFlat)
	alloc.Free(dQrFlat)
	alloc.Free(dKcFlat)
	alloc.Free(dVFlat)
	alloc.Free(dKrFlat)

	// Free forward cache.
	alloc.Free(m.lastX)
	alloc.Free(m.lastQc)
	alloc.Free(m.lastQr)
	alloc.Free(m.lastKc)
	alloc.Free(m.lastKr)
	alloc.Free(m.lastV)
	alloc.Free(m.lastRearranged)
	alloc.Free(m.lastAttn)
	m.lastX = nil
	m.lastQc = nil
	m.lastQr = nil
	m.lastKc = nil
	m.lastKr = nil
	m.lastV = nil
	m.lastRearranged = nil
	m.lastAttn = nil

	if gradOut.dtype != Float32 {
		alloc.Free(gOut)
	}

	return gradIn
}

// Params returns all 7 weight matrices.
func (m *MLA) Params() []optim.Param {
	return []optim.Param{m.W_Q, m.W_DKV, m.W_UK, m.W_UV, m.W_KR, m.W_QR, m.W_O}
}

// freeForwardCache releases the activation cache allocated during Forward.
func (m *MLA) freeForwardCache() {
	if m.lastX != nil {
		alloc.Free(m.lastX)
		alloc.Free(m.lastQc)
		alloc.Free(m.lastQr)
		alloc.Free(m.lastKc)
		alloc.Free(m.lastKr)
		alloc.Free(m.lastV)
		alloc.Free(m.lastRearranged)
		alloc.Free(m.lastAttn)
		m.lastX = nil
		m.lastQc = nil
		m.lastQr = nil
		m.lastKc = nil
		m.lastKr = nil
		m.lastV = nil
		m.lastRearranged = nil
		m.lastAttn = nil
	}
}

// Accessors for serialization.

// WQParam returns a pointer to the W_Q weight.
func (m *MLA) WQParam() *optim.Param { return &m.W_Q }

// WDKVParam returns a pointer to the W_DKV weight.
func (m *MLA) WDKVParam() *optim.Param { return &m.W_DKV }

// WUKParam returns a pointer to the W_UK weight.
func (m *MLA) WUKParam() *optim.Param { return &m.W_UK }

// WUVParam returns a pointer to the W_UV weight.
func (m *MLA) WUVParam() *optim.Param { return &m.W_UV }

// WKRParam returns a pointer to the W_KR weight.
func (m *MLA) WKRParam() *optim.Param { return &m.W_KR }

// WQRParam returns a pointer to the W_QR weight.
func (m *MLA) WQRParam() *optim.Param { return &m.W_QR }

// WOParam returns a pointer to the W_O weight.
func (m *MLA) WOParam() *optim.Param { return &m.W_O }

// DC returns the KV compression dimension.
func (m *MLA) DC() int { return m.dC }

// DHR returns the decoupled RoPE dimension.
func (m *MLA) DHR() int { return m.dHR }

// NumHeads returns the number of attention heads.
func (m *MLA) NumHeads() int { return m.numHeads }

// HeadDim returns the per-head dimension (content).
func (m *MLA) HeadDim() int { return m.headDim }
