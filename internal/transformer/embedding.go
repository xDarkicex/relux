package transformer

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

// Embedding is the token embedding table. Forward looks up
// `weight[tokenID, :]` for each token ID in a [batch, seqLen]
// input. Backward scatters the per-row gradient into the
// corresponding weight rows.
//
// The weight is stored as bfloat16 (per the optim.Param
// contract). The active dtype for the forward pass output is
// float32 — the embedding lookup is a copy, so the output is
// already in float32 (no cast needed).
type Embedding struct {
	BaseModule
	vocabSize int
	dModel    int

	weight optim.Param

	// Forward cache: the input token IDs as a flat int32
	// slice, so Backward can scatter without re-parsing
	// the input.
	lastIDs []int
}

// NewEmbedding constructs an Embedding with vocabSize rows and
// dModel columns. Weight is initialised N(0, 1/sqrt(dModel))
// per the GPT-2 convention.
func NewEmbedding(vocabSize, dModel int) *Embedding {
	if vocabSize <= 0 || dModel <= 0 {
		panic(fmt.Sprintf("NewEmbedding: vocabSize=%d, dModel=%d, both must be > 0", vocabSize, dModel))
	}
	wData := alloc.Uint16(vocabSize * dModel)
	stddev := float32(1.0 / math.Sqrt(float64(dModel)))
	for i := range wData {
		r := float32(((i*1103515245 + 12345) & 0x7fffffff) % 1000) / 500.0 - 1.0
		wData[i] = BF16FromF32(r * stddev)
	}
	return &Embedding{
		vocabSize: vocabSize,
		dModel:    dModel,
		weight: optim.Param{
			Name: "embed.weight",
			Data: wData,
			Grad: alloc.Float32(vocabSize * dModel),
		},
	}
}

// Forward takes flat token IDs and batchSize and returns a
// [batchSize, seqLen, dModel] tensor (seqLen = len(ids)/batchSize).
// Each row [b, s, :] is weight[ids[b*seqLen+s], :] widened bf16→f32.
func (e *Embedding) Forward(ids []int, batchSize int) *Tensor {
	for _, id := range ids {
		if id < 0 || id >= e.vocabSize {
			panic(fmt.Sprintf("Embedding.Forward: token id %d out of range [0, %d)", id, e.vocabSize))
		}
	}
	if len(ids)%batchSize != 0 {
		panic(fmt.Sprintf("Embedding.Forward: len(ids)=%d not divisible by batchSize=%d", len(ids), batchSize))
	}
	seqLen := len(ids) / batchSize
	out := ZerosF32(batchSize, seqLen, e.dModel)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			id := ids[b*seqLen+i]
			src := id * e.dModel
			dst := (b*seqLen + i) * e.dModel
			for j := 0; j < e.dModel; j++ {
				out.DataF32()[dst+j] = F32FromBF16(e.weight.Data[src+j])
			}
		}
	}

	if e.Mode() == Train {
		if e.lastIDs == nil || cap(e.lastIDs) < len(ids) {
			e.lastIDs = make([]int, len(ids))
		}
		e.lastIDs = e.lastIDs[:len(ids)]
		copy(e.lastIDs, ids)
	}

	return out
}

// Backward scatters the per-row gradient into weight.Grad.
// `gradOut` has shape [batchSize, seqLen, dModel].
// For each position [b, s, :], the gradient is added to
// weight.Grad[ids[b*seqLen+s], :].
func (e *Embedding) Backward(gradOut *Tensor) *Tensor {
	if e.lastIDs == nil {
		panic("Embedding.Backward: Forward must be called first (Mode is not Train?)")
	}
	if gradOut.Rank() != 3 || gradOut.shape[2] != e.dModel {
		panic(fmt.Sprintf("Embedding.Backward: shape %v, want [batchSize, seqLen, %d]", gradOut.shape, e.dModel))
	}
	batchSize := gradOut.shape[0]
	seqLen := gradOut.shape[1]
	if batchSize*seqLen != len(e.lastIDs) {
		panic(fmt.Sprintf("Embedding.Backward: gradOut total=%d, Forward saw %d", batchSize*seqLen, len(e.lastIDs)))
	}
	gData, _ := gradOut.ToF32()
	for i, id := range e.lastIDs {
		dst := id * e.dModel
		src := i * e.dModel
		for j := 0; j < e.dModel; j++ {
			e.weight.Grad[dst+j] += gData[src+j]
		}
	}
	return nil
}

// Params returns the embedding weight as an optim.Param.
func (e *Embedding) Params() []optim.Param { return []optim.Param{e.weight} }

// GetParam returns a pointer to the underlying weight Param so
// that callers (e.g. the v1 deserializer) can replace the
// master Data slice. The returned pointer is the same one
// stored in Params() — mutating Data through it is safe.
func (e *Embedding) GetParam() *optim.Param { return &e.weight }
