package transformer

import (
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
)

const (
	// Default block sizes for Flash Attention tiling.
	// Br is the query block size, Bc is the key/value block size.
	// Values 64 or 128 are standard (Dao et al. 2023).
	flashBr = 64
	flashBc = 64
)

// flashOnlineAccum tracks the online softmax state for one query
// row across KV blocks. Equivalent to born's OnlineSoftmax but
// fused for [Br_q, headDim] output in a single accumulator.
type flashOnlineAccum struct {
	maxVal float32   // running max for this query row
	sumExp float32   // running sum of exp(x - max)
	output []float32 // accumulated weighted output [headDim]
}

func newFlashOnlineAccum(headDim int) flashOnlineAccum {
	return flashOnlineAccum{
		maxVal: float32(math.Inf(-1)),
		output: alloc.Float32(headDim),
	}
}

// update processes one KV block for this query row.
// scores[kv] is the attention score for key position kv.
// values is [blockSize * headDim] in row-major order.
func (a *flashOnlineAccum) update(scores, values []float32, blockSize, headDim int) {
	// Find block max.
	blockMax := float32(math.Inf(-1))
	for _, s := range scores {
		if s > blockMax {
			blockMax = s
		}
	}

	oldMax := a.maxVal
	var newMax float32
	if oldMax > blockMax {
		newMax = oldMax
	} else {
		newMax = blockMax
	}

	// Rescale previous accumulations.
	if oldMax != newMax {
		correction := fastexp32(oldMax - newMax)
		a.sumExp *= correction
		for i := range a.output {
			a.output[i] *= correction
		}
	}

	// Add contributions from this block.
	for kv := 0; kv < blockSize; kv++ {
		expScore := fastexp32(scores[kv] - newMax)
		a.sumExp += expScore
		vOff := kv * headDim
		for d := 0; d < headDim; d++ {
			a.output[d] += expScore * values[vOff+d]
		}
	}

	a.maxVal = newMax
}

// normalize divides the accumulated output by the sum.
func (a *flashOnlineAccum) normalize(headDim int) {
	invSum := float32(1.0) / a.sumExp
	for i := 0; i < headDim; i++ {
		a.output[i] *= invSum
	}
}

// free releases the output buffer.
func (a *flashOnlineAccum) free() {
	if a.output != nil {
		alloc.Free(a.output)
		a.output = nil
	}
}

// flashAttentionForward computes attention output using the
// block-tiled Flash Attention 2 algorithm. Never materializes
// the full [seqQ, seqK] attention matrix — memory complexity
// is O(Br × Bc) instead of O(seq²).
//
// q, k, v are [batch * numHeads, seq, headDim] in row-major.
// output is [batch * numHeads, seqQ, headDim] pre-allocated.
// scale is 1/sqrt(headDim). causal applies the upper-tri mask.
func flashAttentionForward(
	q, k, v []float32,
	output []float32,
	batchHeads, seqQ, seqK, headDim int,
	scale float32,
	causal bool,
) {
	headStride := seqQ * headDim // stride between heads in Q/output
	kvStride := seqK * headDim   // stride between heads in K/V

	for bh := 0; bh < batchHeads; bh++ {
		qHead := q[bh*headStride : (bh+1)*headStride]
		kHead := k[bh*kvStride : (bh+1)*kvStride]
		vHead := v[bh*kvStride : (bh+1)*kvStride]
		outHead := output[bh*headStride : (bh+1)*headStride]

		// Tile over query positions.
		for qStart := 0; qStart < seqQ; qStart += flashBr {
			qEnd := qStart + flashBr
			if qEnd > seqQ {
				qEnd = seqQ
			}
			Br := qEnd - qStart
			qBlock := qHead[qStart*headDim : qEnd*headDim]

			// Online softmax accumulators for each query row.
			accums := make([]flashOnlineAccum, Br)
			for i := range accums {
				accums[i] = newFlashOnlineAccum(headDim)
			}

			// Tile over KV positions.
			for kvStart := 0; kvStart < seqK; kvStart += flashBc {
				kvEnd := kvStart + flashBc
				if kvEnd > seqK {
					kvEnd = seqK
				}
				Bc := kvEnd - kvStart
				kBlock := kHead[kvStart*headDim : kvEnd*headDim]
				vBlock := vHead[kvStart*headDim : kvEnd*headDim]

				// S = Q_block @ K_block^T * scale  [Br, Bc]
				scores := alloc.Float32(Br * Bc)
				matmulFloat32TransB(scores, qBlock, kBlock, Br, headDim, Bc)

				// Apply scale and causal mask.
				for i := 0; i < Br; i++ {
					qi := qStart + i
					base := i * Bc
					for j := 0; j < Bc; j++ {
						kj := kvStart + j
						if causal && kj > qi {
							scores[base+j] = float32(math.Inf(-1))
						} else {
							scores[base+j] *= scale
						}
					}
				}

				// Extract V block values as flat [Bc * headDim].
				values := alloc.Float32(Bc * headDim)
				for kv := 0; kv < Bc; kv++ {
					copy(values[kv*headDim:(kv+1)*headDim],
						vBlock[kv*headDim:(kv+1)*headDim])
				}

				// Update online softmax for each query row.
				for i := 0; i < Br; i++ {
					accums[i].update(
						scores[i*Bc:(i+1)*Bc],
						values,
						Bc, headDim,
					)
				}

				alloc.Free(scores)
				alloc.Free(values)
			}

			// Normalize and write output.
			for i := 0; i < Br; i++ {
				accums[i].normalize(headDim)
				qi := qStart + i
				copy(outHead[qi*headDim:(qi+1)*headDim], accums[i].output)
				accums[i].free()
			}
		}
	}
}
