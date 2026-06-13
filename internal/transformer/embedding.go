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

// Forward takes a flat slice of token IDs and returns a
// [seqLen, dModel] tensor (seqLen = len(ids)). Each row i of
// the output is a copy of `weight[ids[i], :]` widened from
// bf16 to float32.
func (e *Embedding) Forward(ids []int) *Tensor {
	for _, id := range ids {
		if id < 0 || id >= e.vocabSize {
			panic(fmt.Sprintf("Embedding.Forward: token id %d out of range [0, %d)", id, e.vocabSize))
		}
	}
	seqLen := len(ids)
	out := ZerosF32(seqLen, e.dModel)
	for i, id := range ids {
		src := id * e.dModel
		dst := i * e.dModel
		for j := 0; j < e.dModel; j++ {
			out.DataF32()[dst+j] = F32FromBF16(e.weight.Data[src+j])
		}
	}

	if e.Mode() == Train {
		if e.lastIDs == nil || cap(e.lastIDs) < seqLen {
			e.lastIDs = make([]int, seqLen)
		}
		e.lastIDs = e.lastIDs[:seqLen]
		copy(e.lastIDs, ids)
	}

	return out
}

// Backward scatters the per-row gradient into weight.Grad.
// `gradOut` has shape [seqLen, dModel] (matching Forward's
// output). For each row i, we add gradOut[i, :] to
// weight.Grad[ids[i], :]. The gradOut is float32 (the
// active dtype); weight.Grad is float32 (per the
// optim.Param.Grad contract).
func (e *Embedding) Backward(gradOut *Tensor) *Tensor {
	if e.lastIDs == nil {
		panic("Embedding.Backward: Forward must be called first (Mode is not Train?)")
	}
	if gradOut.Rank() != 2 || gradOut.shape[1] != e.dModel {
		panic(fmt.Sprintf("Embedding.Backward: shape %v, want [seqLen, %d]", gradOut.shape, e.dModel))
	}
	seqLen := gradOut.shape[0]
	if seqLen != len(e.lastIDs) {
		panic(fmt.Sprintf("Embedding.Backward: gradOut seqLen=%d, Forward saw %d", seqLen, len(e.lastIDs)))
	}
	gData, _ := gradOut.ToF32()
	for i, id := range e.lastIDs {
		dst := id * e.dModel
		src := i * e.dModel
		for j := 0; j < e.dModel; j++ {
			e.weight.Grad[dst+j] += gData[src+j]
		}
	}
	// Embedding has no input gradient (it IS the input).
	return nil
}

// Params returns the embedding weight as an optim.Param.
func (e *Embedding) Params() []optim.Param { return []optim.Param{e.weight} }

// GetParam returns a pointer to the underlying weight Param so
// that callers (e.g. the v1 deserializer) can replace the
// master Data slice. The returned pointer is the same one
// stored in Params() — mutating Data through it is safe.
func (e *Embedding) GetParam() *optim.Param { return &e.weight }
