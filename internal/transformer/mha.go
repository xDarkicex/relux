package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// MHA is Multi-Head Attention with optional Grouped Query
// Attention (when numKVHeads < numHeads; standard MHA when
// equal). Forward is the standard scaled dot-product
// attention with an additive causal mask:
//
//	scores = Q @ K^T / sqrt(headDim)
//	scores = mask ? -inf : scores   (causal upper triangle)
//	weights = softmax(scores, axis=-1)
//	out = weights @ V
//	out = out_proj(out)
//
// where Q, K, V are the linear projections of the input
// (reshaped to [batch, numHeads, seq, headDim] for Q and
// [batch, numKVHeads, seq, headDim] for K/V). GQA expands
// each K/V head to numHeads/numKVHeads query heads.
//
// RoPE is applied to Q and K before the attention score
// computation. The rope module is shared across the model.
//
// For v1, the matmul is pure Go (per-batch/per-head loop).
// The follow-up is to add BatchedMatMul to rnxa for the
// GPU-fast path; the call site is annotated with
// `// TODO(tier3):` for the swap.
type MHA struct {
	BaseModule
	dModel     int
	numHeads   int
	numKVHeads int
	headDim    int
	causal     bool
	rope       *RotaryEmbedding
	dtype      DType

	Wq optim.Param
	Wk optim.Param
	Wv optim.Param
	Wo optim.Param

	Cache          *KVCache // inference-only; nil means no cache
	LayerIdx       int      // which layer this MHA belongs to
	FlashAttention bool     // use block-tiled O(seq) attention

	// Forward cache (Train mode). Set during Forward,
	// consumed by Backward, freed at the end of Backward.
	lastX           []float32 // [batch*seq, dModel]
	lastQ           []float32 // [batch, numHeads, seq, headDim] post-RoPE, post-GQA
	lastK           []float32 // [batch, numHeads, seq, headDim] post-RoPE, post-GQA
	lastV           []float32 // [batch, numHeads, seq, headDim] post-GQA
	lastRearranged  []float32 // [batch*seq, numHeads*headDim] rearrangedBack (pre-Wo)
	lastAttn        []float32 // [batch, numHeads, seq, seq] softmax(weights)
	lastInSeqLen    int
}

// NewMHA constructs a Multi-Head Attention module. rope may
// be nil. causal applies the standard upper-triangular -inf
// mask.
func NewMHA(dModel, numHeads, numKVHeads int, rope *RotaryEmbedding, causal bool) *MHA {
	if dModel <= 0 || numHeads <= 0 || numKVHeads <= 0 {
		panic(fmt.Sprintf("NewMHA: dModel=%d, numHeads=%d, numKVHeads=%d, all must be > 0",
			dModel, numHeads, numKVHeads))
	}
	if dModel%numHeads != 0 {
		panic(fmt.Sprintf("NewMHA: dModel=%d not divisible by numHeads=%d", dModel, numHeads))
	}
	if numHeads%numKVHeads != 0 {
		panic(fmt.Sprintf("NewMHA: numHeads=%d not divisible by numKVHeads=%d (GQA grouping must divide evenly)",
			numHeads, numKVHeads))
	}
	headDim := dModel / numHeads
	if rope != nil && rope.headDim != headDim {
		panic(fmt.Sprintf("NewMHA: rope.headDim=%d != dModel/numHeads=%d", rope.headDim, headDim))
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
	return &MHA{
		dModel:     dModel,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		causal:     causal,
		rope:       rope,
		dtype:      Float32,
		Wq:         mkParam("mha.Wq", initLinearBF16(dModel, numHeads*headDim)),
		Wk:         mkParam("mha.Wk", initLinearBF16(dModel, numKVHeads*headDim)),
		Wv:         mkParam("mha.Wv", initLinearBF16(dModel, numKVHeads*headDim)),
		Wo:         mkParam("mha.Wo", initLinearBF16(numHeads*headDim, dModel)),
	}
}

// Forward takes x of shape [batch, seq, dModel] and returns
// the attention output of shape [batch, seq, dModel].
func (m *MHA) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 {
		panic(fmt.Sprintf("MHA.Forward: input rank=%d, want 3", x.Rank()))
	}
	if x.shape[2] != m.dModel {
		panic(fmt.Sprintf("MHA.Forward: input last dim=%d, want %d", x.shape[2], m.dModel))
	}
	if m.Mode() == Inference && m.Cache != nil {
		return m.forwardCached(x)
	}
	batch := x.shape[0]
	seqQ := x.shape[1]
	seqK := seqQ

	xData, _ := x.ToF32()

	qData := alloc.Float32(batch * seqQ * m.numHeads * m.headDim)
	kData := alloc.Float32(batch * seqQ * m.numKVHeads * m.headDim)
	vData := alloc.Float32(batch * seqQ * m.numKVHeads * m.headDim)
	matmulBatched3D(qData, xData, m.Wq.Data, batch, seqQ, m.dModel, m.numHeads*m.headDim)
	matmulBatched3D(kData, xData, m.Wk.Data, batch, seqQ, m.dModel, m.numKVHeads*m.headDim)
	matmulBatched3D(vData, xData, m.Wv.Data, batch, seqQ, m.dModel, m.numKVHeads*m.headDim)

	qRearranged := rearrangeBSNH(qData, batch, seqQ, m.numHeads, m.headDim)
	kRearranged := rearrangeBSNH(kData, batch, seqQ, m.numKVHeads, m.headDim)
	vRearranged := rearrangeBSNH(vData, batch, seqQ, m.numKVHeads, m.headDim)
	alloc.Free(qData)
	alloc.Free(kData)
	alloc.Free(vData)

	if m.rope != nil {
		qT := &Tensor{shape: []int{batch, m.numHeads, seqQ, m.headDim}, dtype: Float32, f32: qRearranged}
		qRopeT := m.rope.Apply(qT, 0)
		copy(qRearranged, qRopeT.f32)
		alloc.Free(qRopeT.f32)

		kT := &Tensor{shape: []int{batch, m.numKVHeads, seqQ, m.headDim}, dtype: Float32, f32: kRearranged}
		kRopeT := m.rope.Apply(kT, 0)
		copy(kRearranged, kRopeT.f32)
		alloc.Free(kRopeT.f32)
	}

	g := m.numHeads / m.numKVHeads
	if g > 1 {
		kExpanded := alloc.Float32(batch * m.numHeads * seqK * m.headDim)
		vExpanded := alloc.Float32(batch * m.numHeads * seqK * m.headDim)
		for b := 0; b < batch; b++ {
			for h := 0; h < m.numHeads; h++ {
				kvH := h / g
				ksrc := ((b*m.numKVHeads + kvH)*seqK + 0) * m.headDim
				kdst := ((b*m.numHeads + h)*seqK + 0) * m.headDim
				copy(kExpanded[kdst:kdst+seqK*m.headDim], kRearranged[ksrc:ksrc+seqK*m.headDim])
				copy(vExpanded[kdst:kdst+seqK*m.headDim], vRearranged[ksrc:ksrc+seqK*m.headDim])
			}
		}
		alloc.Free(kRearranged)
		alloc.Free(vRearranged)
		kRearranged = kExpanded
		vRearranged = vExpanded
	}

	scale := float32(1.0 / math.Sqrt(float64(m.headDim)))
	attnOut := alloc.Float32(batch * m.numHeads * seqQ * m.headDim)
	var attnWeights []float32

	if m.FlashAttention {
		flashAttentionForward(qRearranged, kRearranged, vRearranged, attnOut,
			batch*m.numHeads, seqQ, seqK, m.headDim, scale, m.causal)
	} else {
		attnWeights = alloc.Float32(batch * m.numHeads * seqQ * seqK)
		for b := 0; b < batch; b++ {
			for h := 0; h < m.numHeads; h++ {
				qOff := (b*m.numHeads + h) * seqQ * m.headDim
				kOff := (b*m.numHeads + h) * seqK * m.headDim
				vOff := (b*m.numHeads + h) * seqK * m.headDim
				wOff := (b*m.numHeads + h) * seqQ * seqK
				oOff := (b*m.numHeads + h) * seqQ * m.headDim
				matmulFloat32TransB(attnWeights[wOff:wOff+seqQ*seqK], qRearranged[qOff:qOff+seqQ*m.headDim], kRearranged[kOff:kOff+seqK*m.headDim], seqQ, m.headDim, seqK)
				for i := 0; i < seqQ; i++ {
					for j := 0; j < seqK; j++ {
						attnWeights[wOff+i*seqK+j] *= scale
						if m.causal && j > i {
							attnWeights[wOff+i*seqK+j] = float32(math.Inf(-1))
						}
					}
				}
				softmaxRows(attnWeights[wOff:wOff+seqQ*seqK], seqQ, seqK)
				matmulFloat32(attnOut[oOff:oOff+seqQ*m.headDim], attnWeights[wOff:wOff+seqQ*seqK], vRearranged[vOff:vOff+seqK*m.headDim], seqQ, seqK, m.headDim)
			}
		}
	}
	alloc.Free(kRearranged)
	alloc.Free(vRearranged)

	rearrangedBack := rearrangeBHSN(attnOut, batch, m.numHeads, seqQ, m.headDim)
	alloc.Free(attnOut)
	out := ZerosF32(batch, seqQ, m.dModel)
	matmulBatched3D(out.DataF32(), rearrangedBack, m.Wo.Data, batch, seqQ, m.numHeads*m.headDim, m.dModel)

	if m.Mode() == Train {
		m.lastX = xData
		m.lastQ = qRearranged
		m.lastK = kRearranged
		m.lastV = vRearranged
		m.lastRearranged = rearrangedBack
		m.lastAttn = attnWeights // nil when FlashAttention is on
		m.lastInSeqLen = seqQ
	} else {
		alloc.Free(xData)
		alloc.Free(qRearranged)
		alloc.Free(kRearranged)
		alloc.Free(vRearranged)
		alloc.Free(rearrangedBack)
		if attnWeights != nil {
			alloc.Free(attnWeights)
		}
	}

	return out
}

// forwardCached is the inference-only forward path with KV-cache.
// When the cache is empty it performs a prefill (compute + cache K/V
// for all input positions). When the cache has entries it performs
// decode (compute K/V for the new position only, append, attend over
// the full cached sequence).
func (m *MHA) forwardCached(x *Tensor) *Tensor {
	batch := x.shape[0]
	seqQ := x.shape[1]
	xData, _ := x.ToF32()

	// 1. Project Q from input (same as standard forward).
	qData := alloc.Float32(batch * seqQ * m.numHeads * m.headDim)
	matmulBatched3D(qData, xData, m.Wq.Data, batch, seqQ, m.dModel, m.numHeads*m.headDim)

	// 2. Project K, V from input.
	kData := alloc.Float32(batch * seqQ * m.numKVHeads * m.headDim)
	vData := alloc.Float32(batch * seqQ * m.numKVHeads * m.headDim)
	matmulBatched3D(kData, xData, m.Wk.Data, batch, seqQ, m.dModel, m.numKVHeads*m.headDim)
	matmulBatched3D(vData, xData, m.Wv.Data, batch, seqQ, m.dModel, m.numKVHeads*m.headDim)

	cacheLen := m.Cache.TotalLen(m.LayerIdx)

	// 3. Rearrange BSNH.
	qRearranged := rearrangeBSNH(qData, batch, seqQ, m.numHeads, m.headDim)
	kRearranged := rearrangeBSNH(kData, batch, seqQ, m.numKVHeads, m.headDim)
	vRearranged := rearrangeBSNH(vData, batch, seqQ, m.numKVHeads, m.headDim)
	alloc.Free(qData)
	alloc.Free(kData)
	alloc.Free(vData)

	// 4. RoPE: Q with offset=cacheLen, K with offset=cacheLen.
	if m.rope != nil {
		qT := &Tensor{shape: []int{batch, m.numHeads, seqQ, m.headDim}, dtype: Float32, f32: qRearranged}
		qRopeT := m.rope.Apply(qT, cacheLen)
		copy(qRearranged, qRopeT.f32)
		alloc.Free(qRopeT.f32)

		kT := &Tensor{shape: []int{batch, m.numKVHeads, seqQ, m.headDim}, dtype: Float32, f32: kRearranged}
		kRopeT := m.rope.Apply(kT, cacheLen)
		copy(kRearranged, kRopeT.f32)
		alloc.Free(kRopeT.f32)
	}

	// 5. Append new K/V to cache (pre-GQA-expand, numKVHeads).
	// The cache stores bf16; Append handles the f32->bf16 cast.
	kCacheT := &Tensor{shape: []int{batch, m.numKVHeads, seqQ, m.headDim}, dtype: Float32, f32: kRearranged}
	vCacheT := &Tensor{shape: []int{batch, m.numKVHeads, seqQ, m.headDim}, dtype: Float32, f32: vRearranged}
	m.Cache.Append(m.LayerIdx, kCacheT, vCacheT)
	alloc.Free(kRearranged)
	alloc.Free(vRearranged)

	// 6. Read full cached K/V → widen bf16→f32.
	fullK, fullV := m.Cache.View(m.LayerIdx)
	// fullK, fullV shape: [batch, numKVHeads, totalLen, headDim]
	cachedKData, _ := fullK.ToF32()
	cachedVData, _ := fullV.ToF32()
	totalLen := fullK.shape[fullK.Rank()-2]

	// 7. GQA expand cached K/V: numKVHeads → numHeads.
	g := m.numHeads / m.numKVHeads
	var kExpanded, vExpanded []float32
	if g > 1 {
		kExpanded = alloc.Float32(batch * m.numHeads * totalLen * m.headDim)
		vExpanded = alloc.Float32(batch * m.numHeads * totalLen * m.headDim)
		for b := 0; b < batch; b++ {
			for h := 0; h < m.numHeads; h++ {
				kvH := h / g
				ksrc := ((b*m.numKVHeads + kvH)*totalLen + 0) * m.headDim
				kdst := ((b*m.numHeads + h)*totalLen + 0) * m.headDim
				copy(kExpanded[kdst:kdst+totalLen*m.headDim], cachedKData[ksrc:ksrc+totalLen*m.headDim])
				copy(vExpanded[kdst:kdst+totalLen*m.headDim], cachedVData[ksrc:ksrc+totalLen*m.headDim])
			}
		}
	} else {
		kExpanded = cachedKData
		vExpanded = cachedVData
	}

	// 8. Attention: Q (seqQ) × K^T (totalLen), softmax, × V.
	scale := float32(1.0 / math.Sqrt(float64(m.headDim)))
	attnOut := alloc.Float32(batch * m.numHeads * seqQ * m.headDim)
	attnWeights := alloc.Float32(batch * m.numHeads * seqQ * totalLen)

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			qOff := (b*m.numHeads + h) * seqQ * m.headDim
			kOff := (b*m.numHeads + h) * totalLen * m.headDim
			vOff := (b*m.numHeads + h) * totalLen * m.headDim
			wOff := (b*m.numHeads + h) * seqQ * totalLen
			oOff := (b*m.numHeads + h) * seqQ * m.headDim

			// Scores = Q @ K^T.
			matmulFloat32TransB(attnWeights[wOff:wOff+seqQ*totalLen], qRearranged[qOff:qOff+seqQ*m.headDim], kExpanded[kOff:kOff+totalLen*m.headDim], seqQ, m.headDim, totalLen)

			for i := 0; i < seqQ; i++ {
				for j := 0; j < totalLen; j++ {
					attnWeights[wOff+i*totalLen+j] *= scale
					// Causal mask: only during prefill (seqQ == totalLen).
					// During decode (seqQ < totalLen), Q is at the end
					// of the sequence and can attend to all cached K.
					if m.causal && seqQ == totalLen && j > i {
						attnWeights[wOff+i*totalLen+j] = float32(math.Inf(-1))
					}
				}
			}
			softmaxRows(attnWeights[wOff:wOff+seqQ*totalLen], seqQ, totalLen)
			matmulFloat32(attnOut[oOff:oOff+seqQ*m.headDim], attnWeights[wOff:wOff+seqQ*totalLen], vExpanded[vOff:vOff+totalLen*m.headDim], seqQ, totalLen, m.headDim)
		}
	}

	if g > 1 {
		alloc.Free(kExpanded)
		alloc.Free(vExpanded)
	}
	alloc.Free(attnWeights)

	// 9. Rearrange back + Wo projection.
	rearrangedBack := rearrangeBHSN(attnOut, batch, m.numHeads, seqQ, m.headDim)
	alloc.Free(attnOut)
	out := ZerosF32(batch, seqQ, m.dModel)
	matmulBatched3D(out.DataF32(), rearrangedBack, m.Wo.Data, batch, seqQ, m.numHeads*m.headDim, m.dModel)

	// Free temporaries.
	alloc.Free(xData)
	alloc.Free(qRearranged)
	alloc.Free(rearrangedBack)
	// cachedKData/cachedVData may alias cache storage; don't free.

	return out
}

// Backward computes the input gradient and accumulates
// into Wq.Grad, Wk.Grad, Wv.Grad, Wo.Grad. gradOut has
// shape [batch, seq, dModel].
//
// The flow reverses the forward. The implementation
// decomposes by per-head; the v1 BatchedMatMul
// follow-up in rnxa collapses the loops. RoPE
// backward is applied via RotaryEmbedding.BackwardApply.
// GQA collapse sums the g duplicate heads back to
// numKVHeads.
func (m *MHA) Backward(gradOut *Tensor) *Tensor {
	if m.lastX == nil {
		panic("MHA.Backward: Forward must be called first (Mode is not Train?)")
	}
	if gradOut.Rank() != 3 || gradOut.shape[2] != m.dModel {
		panic(fmt.Sprintf("MHA.Backward: gradOut shape %v, want [batch, seq, %d]", gradOut.shape, m.dModel))
	}
	batch := gradOut.shape[0]
	seq := gradOut.shape[1]
	n := batch * seq
	headDim := m.headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	gOut, _ := gradOut.ToF32()
	xData := m.lastX
	wqData := make([]float32, len(m.Wq.Data))
	wkData := make([]float32, len(m.Wk.Data))
	wvData := make([]float32, len(m.Wv.Data))
	woData := make([]float32, len(m.Wo.Data))
	for i, w := range m.Wq.Data {
		wqData[i] = F32FromBF16(w)
	}
	for i, w := range m.Wk.Data {
		wkData[i] = F32FromBF16(w)
	}
	for i, w := range m.Wv.Data {
		wvData[i] = F32FromBF16(w)
	}
	for i, w := range m.Wo.Data {
		woData[i] = F32FromBF16(w)
	}

	// Step 1-3: dL/dRearrangedBack = gradOut @ Wo^T.
	// Wo is [numHeads*headDim, dModel]. Wo^T is [dModel, numHeads*headDim].
	// dRearranged[i, j] = sum_k gradOut[i, k] * Wo[j, k]
	dRearranged := alloc.Float32(n * m.numHeads * headDim)
	for i := 0; i < n; i++ {
		for j := 0; j < m.numHeads*headDim; j++ {
			var sum float32
			for k := 0; k < m.dModel; k++ {
				sum += gOut[i*m.dModel+k] * woData[j*m.dModel+k]
			}
			dRearranged[i*m.numHeads*headDim+j] = sum
		}
	}

	// Wo.Grad += sum_n rearrangedBack[n] outer gradOut[n].
	for j := 0; j < m.numHeads*headDim; j++ {
		for k := 0; k < m.dModel; k++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += m.lastRearranged[i*m.numHeads*headDim+j] * gOut[i*m.dModel+k]
			}
			m.Wo.Grad[j*m.dModel+k] += sum
		}
	}

	// Step 4: dL/dAttnOut = rearrangeBSNH(dRearranged).
	dAttnOut := rearrangeBSNH(dRearranged, batch, m.numHeads, seq, headDim)
	alloc.Free(dRearranged)

	// Step 5+6: dL/dAttnWeights and dL/dV_postGQA, per head.
	// dAttnW[b, h, i, j] = sum_k dAttnOut[b, h, i, k] * V[b, h, j, k]
	// dV[b, h, j, k] = sum_i attnW[b, h, i, j] * dAttnOut[b, h, i, k]
	dAttnW := alloc.Float32(batch * m.numHeads * seq * seq)
	dV_postGQA := alloc.Float32(batch * m.numHeads * seq * headDim)

	// Recompute attention weights if flash attention was used
	// (lastAttn is nil — forward used the O(seq) path).
	if m.lastAttn == nil {
		m.lastAttn = alloc.Float32(batch * m.numHeads * seq * seq)
		for b := 0; b < batch; b++ {
			for h := 0; h < m.numHeads; h++ {
				qOff := (b*m.numHeads + h) * seq * m.headDim
				kOff := (b*m.numHeads + h) * seq * m.headDim
				wOff := (b*m.numHeads + h) * seq * seq
				matmulFloat32TransB(m.lastAttn[wOff:wOff+seq*seq], m.lastQ[qOff:qOff+seq*m.headDim], m.lastK[kOff:kOff+seq*m.headDim], seq, m.headDim, seq)
				for i := 0; i < seq; i++ {
					for j := 0; j < seq; j++ {
						m.lastAttn[wOff+i*seq+j] *= scale
						if m.causal && j > i {
							m.lastAttn[wOff+i*seq+j] = float32(math.Inf(-1))
						}
					}
				}
				softmaxRows(m.lastAttn[wOff:wOff+seq*seq], seq, seq)
			}
		}
	}

	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			dAOff := (b*m.numHeads + h) * seq * headDim
			dVOff := (b*m.numHeads + h) * seq * headDim
			dWOff := (b*m.numHeads + h) * seq * seq
			aOff := (b*m.numHeads + h) * seq * seq
			vOff := (b*m.numHeads + h) * seq * headDim
			// Pass 1: dAttnW.
			for i := 0; i < seq; i++ {
				for j := 0; j < seq; j++ {
					var sum float32
					for k := 0; k < headDim; k++ {
						sum += dAttnOut[dAOff+i*headDim+k] * m.lastV[vOff+j*headDim+k]
					}
					dAttnW[dWOff+i*seq+j] = sum
				}
			}
			// Pass 2: dV.
			for j := 0; j < seq; j++ {
				for k := 0; k < headDim; k++ {
					var sum float32
					for i := 0; i < seq; i++ {
						sum += m.lastAttn[aOff+i*seq+j] * dAttnOut[dAOff+i*headDim+k]
					}
					dV_postGQA[dVOff+j*headDim+k] = sum
				}
			}
		}
	}
	alloc.Free(dAttnOut)

	// Step 7: softmax backward, per row.
	// dScores[i, j] = attnW[i, j] * (dAttnW[i, j] - sum_k dAttnW[i, k] * attnW[i, k])
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

	// Step 8: dL/dQ, dL/dK post-GQA from dScores. We do both
	// in a single loop so dScores is used once. Both
	// post-GQA, post-RoPE.
	// dQ[b, h, i, k] = sum_j dScores[b, h, i, j] * K[b, h, j, k] / sqrt(d_k)
	// dK[b, h, j, k] = sum_i dScores[b, h, i, j] * Q[b, h, i, k] / sqrt(d_k)
	dQ := alloc.Float32(batch * m.numHeads * seq * headDim)
	dK := alloc.Float32(batch * m.numHeads * seq * headDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < m.numHeads; h++ {
			dQOff := (b*m.numHeads + h) * seq * headDim
			dKOff := (b*m.numHeads + h) * seq * headDim
			qOff := (b*m.numHeads + h) * seq * headDim
			kOff := (b*m.numHeads + h) * seq * headDim
			sOff := (b*m.numHeads + h) * seq * seq
			// Pass 1: dQ. For each (i, k), sum over j.
			for i := 0; i < seq; i++ {
				for k := 0; k < headDim; k++ {
					var sum float32
					for j := 0; j < seq; j++ {
						sum += dScores[sOff+i*seq+j] * m.lastK[kOff+j*headDim+k]
					}
					dQ[dQOff+i*headDim+k] = sum * scale
				}
			}
			// Pass 2: dK. For each (j, k), sum over i.
			for j := 0; j < seq; j++ {
				for k := 0; k < headDim; k++ {
					var sum float32
					for i := 0; i < seq; i++ {
						sum += dScores[sOff+i*seq+j] * m.lastQ[qOff+i*headDim+k]
					}
					dK[dKOff+j*headDim+k] = sum * scale
				}
			}
		}
	}
	alloc.Free(dScores)

	// Step 9: RoPE backward (inverse rotation).
	if m.rope != nil {
		dQT := &Tensor{shape: []int{batch, m.numHeads, seq, headDim}, dtype: Float32, f32: dQ}
		dQRope := m.rope.BackwardApply(dQT, 0)
		copy(dQ, dQRope.f32)
		alloc.Free(dQRope.f32)
		dKT := &Tensor{shape: []int{batch, m.numHeads, seq, headDim}, dtype: Float32, f32: dK}
		dKRope := m.rope.BackwardApply(dKT, 0)
		copy(dK, dKRope.f32)
		alloc.Free(dKRope.f32)
	}

	// Step 10: GQA collapse. Sum the g duplicate heads
	// back to numKVHeads. dQ, dK, dV_postGQA are all
	// [batch, numHeads, seq, headDim]. The collapsed
	// versions are [batch, numKVHeads, seq, headDim].
	//
	// Note: dQ is used for the Wq gradient (Wq has
	// numHeads*headDim columns) — we do NOT collapse dQ
	// (the Wq gradient is per-query-head, not per-KV-head).
	// dK and dV are collapsed to numKVHeads because Wk and
	// Wv are numKVHeads*headDim wide.
	g := m.numHeads / m.numKVHeads
	dK_preGQA := alloc.Float32(batch * m.numKVHeads * seq * headDim)
	dV_preGQA := alloc.Float32(batch * m.numKVHeads * seq * headDim)
	if g == 1 {
		// No GQA. Just rename dK -> dK_preGQA, etc.
		dK_preGQA = dK
		dV_preGQA = dV_postGQA
	} else {
		for b := 0; b < batch; b++ {
			for kvH := 0; kvH < m.numKVHeads; kvH++ {
				for j := 0; j < seq; j++ {
					for k := 0; k < headDim; k++ {
						var sumK, sumV float32
						for qh := 0; qh < g; qh++ {
							h := kvH*g + qh
							sumK += dK[(b*m.numHeads+h)*seq*headDim+j*headDim+k]
							sumV += dV_postGQA[(b*m.numHeads+h)*seq*headDim+j*headDim+k]
						}
						dK_preGQA[(b*m.numKVHeads+kvH)*seq*headDim+j*headDim+k] = sumK
						dV_preGQA[(b*m.numKVHeads+kvH)*seq*headDim+j*headDim+k] = sumV
					}
				}
			}
		}
		alloc.Free(dK)
		alloc.Free(dV_postGQA)
	}
	// dQ is still [batch, numHeads, seq, headDim] (numHeads
	// wide; used for Wq.Grad which is numHeads*headDim wide).
	// dK_preGQA, dV_preGQA are [batch, numKVHeads, seq,
	// headDim] (numKVHeads wide; used for Wk.Grad and
	// Wv.Grad which are numKVHeads*headDim wide).

	// Rearrange to [batch*seq, numHeads*headDim] for dQ
	// and [batch*seq, numKVHeads*headDim] for dK, dV.
	dQ_rearranged := rearrangeBHSN(dQ, batch, m.numHeads, seq, headDim)
	dK_rearranged := rearrangeBHSN(dK_preGQA, batch, m.numKVHeads, seq, headDim)
	dV_rearranged := rearrangeBHSN(dV_preGQA, batch, m.numKVHeads, seq, headDim)
	alloc.Free(dQ)
	alloc.Free(dK_preGQA)
	alloc.Free(dV_preGQA)

	// Step 12: Wq.Grad += x^T @ dQ_rearranged (uses the
	// numHeads-wide dQ).
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.numHeads*headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dQ_rearranged[i*m.numHeads*headDim+j]
			}
			m.Wq.Grad[r*m.numHeads*headDim+j] += sum
		}
	}
	// Wk.Grad, Wv.Grad use the numKVHeads-wide dK, dV.
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.numKVHeads*headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dK_rearranged[i*m.numKVHeads*headDim+j]
			}
			m.Wk.Grad[r*m.numKVHeads*headDim+j] += sum
		}
	}
	for r := 0; r < m.dModel; r++ {
		for j := 0; j < m.numKVHeads*headDim; j++ {
			var sum float32
			for i := 0; i < n; i++ {
				sum += xData[i*m.dModel+r] * dV_rearranged[i*m.numKVHeads*headDim+j]
			}
			m.Wv.Grad[r*m.numKVHeads*headDim+j] += sum
		}
	}

	// Step 13: dL/dX = dQ_rearranged @ Wq^T +
	//                   dK_rearranged @ Wk^T +
	//                   dV_rearranged @ Wv^T.
	gradIn := ZerosF32(batch, seq, m.dModel)
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.numHeads*headDim; j++ {
				sum += dQ_rearranged[i*m.numHeads*headDim+j] * wqData[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.numKVHeads*headDim; j++ {
				sum += dK_rearranged[i*m.numKVHeads*headDim+j] * wkData[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}
	for i := 0; i < n; i++ {
		for r := 0; r < m.dModel; r++ {
			var sum float32
			for j := 0; j < m.numKVHeads*headDim; j++ {
				sum += dV_rearranged[i*m.numKVHeads*headDim+j] * wvData[j*m.dModel+r]
			}
			gradIn.DataF32()[i*m.dModel+r] += sum
		}
	}

	// Free all temporaries.
	alloc.Free(dQ_rearranged)
	alloc.Free(dK_rearranged)
	alloc.Free(dV_rearranged)
	alloc.Free(m.lastX)
	alloc.Free(m.lastQ)
	alloc.Free(m.lastK)
	alloc.Free(m.lastV)
	alloc.Free(m.lastRearranged)
	alloc.Free(m.lastAttn)
	m.lastX = nil
	m.lastQ = nil
	m.lastK = nil
	m.lastV = nil
	m.lastRearranged = nil
	m.lastAttn = nil

	if gradOut.dtype != Float32 {
		alloc.Free(gOut)
	}

	return gradIn
}

// Params returns the four linear projections.
// freeForwardCache releases the activation cache allocated
// during Forward. Used by gradient checkpointing.
func (m *MHA) freeForwardCache() {
	if m.lastX != nil {
		alloc.Free(m.lastX)
		alloc.Free(m.lastQ)
		alloc.Free(m.lastK)
		alloc.Free(m.lastV)
		alloc.Free(m.lastRearranged)
		if m.lastAttn != nil {
			alloc.Free(m.lastAttn)
		}
		m.lastX = nil
		m.lastQ = nil
		m.lastK = nil
		m.lastV = nil
		m.lastRearranged = nil
		m.lastAttn = nil
	}
}

func (m *MHA) Params() []optim.Param {
	return []optim.Param{m.Wq, m.Wk, m.Wv, m.Wo}
}

// WqParam returns a pointer to the Wq weight.
func (m *MHA) WqParam() *optim.Param { return &m.Wq }

// WkParam returns a pointer to the Wk weight.
func (m *MHA) WkParam() *optim.Param { return &m.Wk }

// WvParam returns a pointer to the Wv weight.
func (m *MHA) WvParam() *optim.Param { return &m.Wv }

// WoParam returns a pointer to the Wo weight.
func (m *MHA) WoParam() *optim.Param { return &m.Wo }

// matmulBatched3D computes C = A @ B where A is the
// concatenation of `batch` matrices each of shape [seq, in],
// B is [in, out] stored as bfloat16, and C is the
// concatenation of `batch` matrices each of shape [seq, out].
// A and C are flat [batch*seq, in] and [batch*seq, out]
// float32. B is widened from bf16 to f32 on the fly.
//
// When Backend is set, the matmul is dispatched through the
// compute backend (rnxa / MPS / Metal / CUDA) for hardware
// acceleration. The bf16→f32 widening happens once per call
// (not per multiply) to minimize the CPU→GPU transfer path.
func matmulBatched3D(out, a []float32, bData []uint16, batch, seq, inDim, outDim int) {
	M := batch * seq
	if Backend != nil && Backend.Available() {
		// Widen bf16 weights to f32 once per call.
		bF32 := make([]float32, len(bData))
		for i, b := range bData {
			bF32[i] = F32FromBF16(b)
		}
		result, err := Backend.MatMulFloat32(a, bF32, M, inDim, outDim)
		if err == nil {
			copy(out, result)
			return
		}
		// On error, fall through to pure Go.
	}
	for i := 0; i < M; i++ {
		aRow := i * inDim
		outRow := i * outDim
		for j := 0; j < outDim; j++ {
			var sum float32
			for k := 0; k < inDim; k++ {
				sum += a[aRow+k] * F32FromBF16(bData[k*outDim+j])
			}
			out[outRow+j] = sum
		}
	}
}

// rearrangeBSNH reorders [batch, seq, numHeads, headDim] ->
// [batch, numHeads, seq, headDim].
func rearrangeBSNH(src []float32, batch, seq, numHeads, headDim int) []float32 {
	out := alloc.Float32(batch * numHeads * seq * headDim)
	rowSize := numHeads * headDim
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			srcRow := b*seq*rowSize + s*rowSize
			for h := 0; h < numHeads; h++ {
				srcOff := srcRow + h*headDim
				dstOff := (b*numHeads+h)*seq*headDim + s*headDim
				copy(out[dstOff:dstOff+headDim], src[srcOff:srcOff+headDim])
			}
		}
	}
	return out
}

// rearrangeBHSN reorders [batch, numHeads, seq, headDim] ->
// [batch, seq, numHeads*headDim]. Inverse of rearrangeBSNH.
func rearrangeBHSN(src []float32, batch, numHeads, seq, headDim int) []float32 {
	out := alloc.Float32(batch * seq * numHeads * headDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seq; s++ {
				srcOff := (b*numHeads+h)*seq*headDim + s*headDim
				dstOff := b*seq*numHeads*headDim + s*numHeads*headDim + h*headDim
				copy(out[dstOff:dstOff+headDim], src[srcOff:srcOff+headDim])
			}
		}
	}
	return out
}

// fastexp32 computes exp(x) using a pure-float32 bit-level
// approximation. Accurate to ~2% relative error for |x| < 20.
func fastexp32(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 87 {
		return math.Float32frombits(0x7F800000) // +Inf
	}
	const factor float32 = float32(1<<23) / 0.6931471805599453
	const bias float32 = 127.0*float32(1<<23) - 405000
	return math.Float32frombits(uint32(int32(factor*x + bias)))
}

// softmaxRows applies softmax row-by-row to a [rows, cols]
// matrix in place, using a fast float32 exp approximation.
func softmaxRows(m []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := m[r*cols : (r+1)*cols]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		var sum float32
		for j, v := range row {
			diff := v - maxV
			if diff < -20 {
				row[j] = 0
			} else {
				row[j] = fastexp32(diff)
			}
			sum += row[j]
		}
		invSum := float32(1.0) / sum
		for j := range row {
			row[j] *= invSum
		}
	}
}
