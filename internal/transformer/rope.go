package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
)

// RotaryEmbedding is the rotary positional embedding used by
// LLaMA, Mistral, Qwen, MiniMax, and most modern LLMs. The
// rotation is applied in pairs over the last dim of the input
// (Q or K, headDim-dimensional), so headDim must be even.
//
// The angular frequency for pair index i in [0, headDim/2) is:
//
//	theta_i = base^(-2i / headDim)
//
// The standard base is 10000. Long-context variants like
// MiniMax-Text-01 use 10,000,000 for extended range. NTK-aware
// scaling (e.g. RoPE-Yarn) can also be approximated by passing
// a per-position scaling factor; for v1 we just precompute
// the standard cos/sin tables.
//
// Apply rotates Q or K:
//
//	(x_{2i}, x_{2i+1}) -> (x_{2i}*cos[m,i] - x_{2i+1}*sin[m,i],
//	                       x_{2i+1}*cos[m,i] + x_{2i}*sin[m,i])
//
// where m is the position. startPos lets the caller offset
// for KV-cache inference (position 0 in a chunked input is
// position startPos in the full sequence).
type RotaryEmbedding struct {
	BaseModule
	headDim    int
	base       float32
	maxSeqLen  int

	// Precomputed cos/sin tables: shape [maxSeqLen, headDim/2].
	// cos and sin are float32 (the precompute is cheap; we
	// don't store bf16 here because it's used per-call in the
	// inner loop).
	cos []float32
	sin []float32
}

// NewRotaryEmbedding precomputes the cos/sin tables up to
// maxSeqLen for the given headDim and base. headDim must be
// even. panics on invalid input.
func NewRotaryEmbedding(headDim int, base float32, maxSeqLen int) *RotaryEmbedding {
	if headDim <= 0 || headDim%2 != 0 {
		panic(fmt.Sprintf("NewRotaryEmbedding: headDim=%d, must be even and > 0", headDim))
	}
	if maxSeqLen <= 0 {
		panic(fmt.Sprintf("NewRotaryEmbedding: maxSeqLen=%d, must be > 0", maxSeqLen))
	}
	half := headDim / 2
	cosT := alloc.Float32(maxSeqLen * half)
	sinT := alloc.Float32(maxSeqLen * half)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < half; i++ {
			theta := float32(1.0) / float32(math.Pow(float64(base), float64(2*i)/float64(headDim)))
			angle := float32(float64(pos) * float64(theta))
			cosT[pos*half+i] = float32(math.Cos(float64(angle)))
			sinT[pos*half+i] = float32(math.Sin(float64(angle)))
		}
	}
	return &RotaryEmbedding{
		headDim:   headDim,
		base:      base,
		maxSeqLen: maxSeqLen,
		cos:       cosT,
		sin:       sinT,
	}
}

// Cos returns the precomputed cos table. Exposed for tests
// and for diagnostic output; not part of the Module interface.
func (r *RotaryEmbedding) Cos() []float32 { return r.cos }

// Sin returns the precomputed sin table.
func (r *RotaryEmbedding) Sin() []float32 { return r.sin }

// Apply rotates the last dim of x in pairs. The input is
// the Q or K tensor from MHA, shape [batch, numHeads,
// seqLen, headDim] (4D) or just [seqLen, headDim] (2D).
// startPos is the position offset (0 for prefill, the
// current cache length for decode).
//
// The rotation uses the *position* within the sequence, not
// the flat row index. With [batch, numHeads, seq, headDim],
// the position for row r is `r % seq` (the layout is
// [b*numHeads*seq + h*seq + s] for the row index).
func (r *RotaryEmbedding) Apply(x *Tensor, startPos int) *Tensor {
	if x.Rank() == 0 || x.shape[len(x.shape)-1] != r.headDim {
		panic(fmt.Sprintf("RotaryEmbedding.Apply: last dim = %d, want %d",
			x.shape[len(x.shape)-1], r.headDim))
	}
	half := r.headDim / 2
	// For 1D input the whole tensor is the sequence; for
	// 2D+ the seq axis is at position Rank-2.
	var seqLen int
	if x.Rank() == 1 {
		seqLen = 1 // only one row, position is startPos (or 0)
	} else {
		seqLen = x.shape[x.Rank()-2]
	}

	xData, _ := x.ToF32()
	out := ZerosF32(x.shape...)
	rows := x.Size() / r.headDim

	for row := 0; row < rows; row++ {
		base := row * r.headDim
		var posIdx int
		if startPos > 0 {
			posIdx = startPos * half
		} else if x.Rank() == 1 {
			posIdx = 0
		} else {
			pos := row % seqLen
			posIdx = pos * half
		}
		cosBase := r.cos[posIdx : posIdx+half]
		sinBase := r.sin[posIdx : posIdx+half]

		for i := 0; i < half; i++ {
			x0 := xData[base+2*i]
			x1 := xData[base+2*i+1]
			c := cosBase[i]
			s := sinBase[i]
			out.DataF32()[base+2*i] = x0*c - x1*s
			out.DataF32()[base+2*i+1] = x1*c + x0*s
		}
	}

	if x.dtype != Float32 {
		alloc.Free(xData)
	}
	return out
}

// BackwardApply is the inverse rotation. Because the rotation
// is orthogonal (cos^2 + sin^2 = 1), the backward is the
// same rotation with the sign of sin flipped.
func (r *RotaryEmbedding) BackwardApply(gradOut *Tensor, startPos int) *Tensor {
	if gradOut.Rank() == 0 || gradOut.shape[len(gradOut.shape)-1] != r.headDim {
		panic(fmt.Sprintf("RotaryEmbedding.BackwardApply: last dim = %d, want %d",
			gradOut.shape[len(gradOut.shape)-1], r.headDim))
	}
	half := r.headDim / 2
	var seqLen int
	if gradOut.Rank() == 1 {
		seqLen = 1
	} else {
		seqLen = gradOut.shape[gradOut.Rank()-2]
	}

	gData, _ := gradOut.ToF32()
	out := ZerosF32(gradOut.shape...)
	rows := gradOut.Size() / r.headDim

	for row := 0; row < rows; row++ {
		base := row * r.headDim
		var posIdx int
		if startPos > 0 {
			posIdx = startPos * half
		} else if gradOut.Rank() == 1 {
			posIdx = 0
		} else {
			pos := row % seqLen
			posIdx = pos * half
		}
		cosBase := r.cos[posIdx : posIdx+half]
		sinBase := r.sin[posIdx : posIdx+half]

		for i := 0; i < half; i++ {
			g0 := gData[base+2*i]
			g1 := gData[base+2*i+1]
			c := cosBase[i]
			s := sinBase[i]
			out.DataF32()[base+2*i] = g0*c + g1*s
			out.DataF32()[base+2*i+1] = g1*c - g0*s
		}
	}

	if gradOut.dtype != Float32 {
		alloc.Free(gData)
	}
	return out
}
