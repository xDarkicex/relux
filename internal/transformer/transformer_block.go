package transformer

import (
	"fmt"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
)

const (
	// ATTN_MHA is standard multi-head attention.
	ATTN_MHA = 0
	// ATTN_MLA is multi-head latent attention with compressed KV cache.
	ATTN_MLA = 1
)

// Block is a single transformer block. The pre-norm
// composition (LLaMA-style) is:
//
//	h = x + MHA(RMSNorm(x), rope, causal)
//	y = h + MLP(RMSNorm(h))
//
// Each block owns two RMSNorms (one before MHA, one before
// MLP). The MHA receives a shared rope module; in practice
// each block holds a reference to the same rope.
//
// The block's Params() returns all trainable parameters:
// MHA's 4 projections + MLP's 4 + 2 RMSNorm gammas.
//
// Gradient checkpointing: when checkpoint is true, Forward
// frees submodule activation caches immediately, keeping only
// lastX (the block input). Backward re-runs the submodule
// forward passes to regenerate the caches before computing
// gradients. Memory per block drops from O(seq² + seq×dFF)
// to O(seq×dModel).
type Block struct {
	BaseModule
	normAttn   *RMSNorm
	mha        *MHA
	mla        *MLA
	normMlp    *RMSNorm
	mlp        *MLP
	attnType   int
	checkpoint bool

	// Forward cache for the residual stream (used by
	// Backward to compute the residual connection's
	// gradient). lastX is the block's input; lastH is
	// the residual stream after the MHA.
	lastX *Tensor
	lastH *Tensor
}

// NewBlock constructs a single transformer block. rope is
// shared across all blocks (one RoPE module per model).
// ffnType selects the feedforward variant (FFNGELU or FFNSwiGLU).
// attnType is ATTN_MHA or ATTN_MLA.
// For MLA: dC is the KV compression dim, dHR is the RoPE dim.
// maxSeqLen and ropeBase are extracted from rope for MLA's decoupled RoPE.
func NewBlock(dModel, numHeads, numKVHeads, dFF int, rope *RotaryEmbedding, causal bool, ffnType FFNType, checkpoint bool, flashAttention bool, attnType int, dC int, dHR int) *Block {
	b := &Block{
		normAttn:   NewRMSNorm(dModel, 1e-5),
		normMlp:    NewRMSNorm(dModel, 1e-5),
		mlp:        NewMLP(dModel, dFF, ffnType),
		attnType:   attnType,
		checkpoint: checkpoint,
	}

	if attnType == ATTN_MLA {
		maxSeqLen := 2048
		ropeBase := float32(10000)
		if rope != nil {
			maxSeqLen = rope.MaxSeqLen()
		}
		c := NewMLA(dModel, numHeads, dC, dHR, maxSeqLen, ropeBase, causal)
		c.FlashAttention = flashAttention
		b.mla = c
	} else {
		mha := NewMHA(dModel, numHeads, numKVHeads, rope, causal)
		mha.FlashAttention = flashAttention
		b.mha = mha
	}
	return b
}

// attnForward returns the attention module's Forward output.
func (b *Block) attnForward(x *Tensor) *Tensor {
	if b.attnType == ATTN_MLA {
		return b.mla.Forward(x)
	}
	return b.mha.Forward(x)
}

// attnCacheFreed reports whether the attention module's forward cache
// has been freed (nil lastX on the attention module).
func (b *Block) attnCacheFreed() bool {
	if b.attnType == ATTN_MLA {
		return b.mla.lastX == nil
	}
	return b.mha.lastX == nil
}

// attnFreeForwardCache frees the attention module's forward cache.
func (b *Block) attnFreeForwardCache() {
	if b.attnType == ATTN_MLA {
		b.mla.freeForwardCache()
	} else {
		b.mha.freeForwardCache()
	}
}

// Forward computes the block output. Input shape [batch,
// seq, dModel] -> output shape [batch, seq, dModel].
//
// When gradient checkpointing is enabled, submodule activation
// caches (attention scores, MLP hidden states) are freed after
// the forward pass; only lastX is retained. Backward will
// re-run the submodule forwards to regenerate the caches.
func (b *Block) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 || x.shape[2] != 0 {
		_ = 0
	}
	b.lastX = x.Clone()
	b.lastH = x

	normed1 := b.normAttn.Forward(x)
	attnOut := b.attnForward(normed1)
	postAttn := residualAdd(x, attnOut)

	normed2 := b.normMlp.Forward(postAttn)
	mlpOut := b.mlp.Forward(normed2)
	out := residualAdd(postAttn, mlpOut)

	if b.checkpoint {
		b.normAttn.freeForwardCache()
		b.attnFreeForwardCache()
		b.normMlp.freeForwardCache()
		b.mlp.freeForwardCache()
	}

	return out
}

// recomputeForward re-runs the submodule forward passes from
// the saved block input to regenerate activation caches.
func (b *Block) recomputeForward() {
	x := b.lastX
	normed1 := b.normAttn.Forward(x)
	attnOut := b.attnForward(normed1)
	postAttn := residualAdd(x, attnOut)
	normed2 := b.normMlp.Forward(postAttn)
	b.mlp.Forward(normed2)
}

// Backward computes the input gradient. The flow is the
// reverse of Forward:
//
//	dL/dOut = gradOut
//	postAttn_grad = dL/dOut (from the second residual add)
//	postAttn_grad = dL/dOut (the MLP residual is the same
//	  stream as the MHA's output)
//	dL/dMLP_in, dL/dMLP_W = MLP.Backward(postAttn_grad)
//	preMlp_grad = postAttn_grad + dL/dMLP_in
//	dL/dNormMlp_gamma += norm backward
//	dL/dH_postAttn = preMlp_grad (which is the input to MLP)
//	dL/dAttn_in, dL/dAttn_W = MHA.Backward(dL/dH_postAttn)
//	preAttn_grad = dL/dH_postAttn + dL/dAttn_in
//	dL/dNormAttn_gamma += norm backward
//	dL/dX = preAttn_grad (the block's input)
func (b *Block) Backward(gradOut *Tensor) *Tensor {
	if b.lastX == nil {
		panic("Block.Backward: Forward must be called first (Mode is not Train?)")
	}

	// Recompute submodule forward passes if checkpointing
	// freed the activation caches.
	if b.checkpoint && b.attnCacheFreed() {
		b.recomputeForward()
	}

	// Split the gradient through the second residual add.
	// The MLP input was `postAttn = x + attnOut`. The MLP
	// output is `out = postAttn + mlpOut`. dL/d(postAttn)
	// = gradOut (residual) + dL/d(postAttn) (through the
	// MLP path: normMlp backward applied to mlpInGrad).
	mlpInGrad := b.mlp.Backward(gradOut)
	normMlpInGrad := b.normMlp.Backward(mlpInGrad)
	postAttnGrad := residualAddGrad(gradOut, normMlpInGrad)
	// Now postAttnGrad = dL/d(postAttn).
	// The attention residual add was x + attnOut. dL/d(attn_in)
	// = dL/d(postAttn) -> attn.Backward.
	var attnInGrad *Tensor
	if b.attnType == ATTN_MLA {
		attnInGrad = b.mla.Backward(postAttnGrad)
	} else {
		attnInGrad = b.mha.Backward(postAttnGrad)
	}
	// attnInGrad = dL/d(normed1). dL/d(x) = residual +
	// normAttn backward of attnInGrad.
	normAttnInGrad := b.normAttn.Backward(attnInGrad)
	gradIn := residualAddGrad(postAttnGrad, normAttnInGrad)

	// Free the block-level cache. In the checkpoint path,
	// recomputeForward passes b.lastX to normAttn, whose
	// Backward frees the aliased data. In the non-checkpoint
	// path, b.lastH is aliased by normAttn.lastX (also freed
	// by normAttn.Backward). Only b.lastX in the
	// non-checkpoint path needs explicit cleanup.
	if b.checkpoint {
		b.lastX = nil
	} else if b.lastX != nil {
		allocFreeTensor(b.lastX)
		b.lastX = nil
	}
	b.lastH = nil

	return gradIn
}

// Params returns all trainable parameters: attention params +
// MLP params + 2 RMSNorm gammas.
func (b *Block) Params() []optim.Param {
	out := []optim.Param{}
	out = append(out, b.normAttn.Params()...)
	if b.attnType == ATTN_MLA {
		out = append(out, b.mla.Params()...)
	} else {
		out = append(out, b.mha.Params()...)
	}
	out = append(out, b.normMlp.Params()...)
	out = append(out, b.mlp.Params()...)
	return out
}

// NormAttn returns the pre-attention RMSNorm.
func (b *Block) NormAttn() *RMSNorm { return b.normAttn }

// BlockMHA returns the multi-head attention module. Returns nil when
// the block uses MLA.
func (b *Block) BlockMHA() *MHA { return b.mha }

// BlockMLA returns the multi-head latent attention module. Returns nil
// when the block uses standard MHA.
func (b *Block) BlockMLA() *MLA { return b.mla }

// AttnType returns the attention type (ATTN_MHA or ATTN_MLA).
func (b *Block) AttnType() int { return b.attnType }

// NormMlp returns the pre-MLP RMSNorm.
func (b *Block) NormMlp() *RMSNorm { return b.normMlp }

// BlockMLP returns the feedforward MLP module.
func (b *Block) BlockMLP() *MLP { return b.mlp }

// residualAdd returns a + b. a and b must have the same
// shape. The result is a fresh allocation.
func residualAdd(a, b *Tensor) *Tensor {
	if !sameShape(a.shape, b.shape) {
		panic(fmt.Sprintf("residualAdd: shape mismatch %v vs %v", a.shape, b.shape))
	}
	out := ZerosF32(a.shape...)
	aData, _ := a.ToF32()
	bData, _ := b.ToF32()
	for i := range aData {
		out.DataF32()[i] = aData[i] + bData[i]
	}
	if a.dtype != Float32 {
		alloc.Free(aData)
	}
	if b.dtype != Float32 {
		alloc.Free(bData)
	}
	return out
}

// residualAddGrad returns gradA + gradB (both gradA and
// gradB have the same shape). For the residual connection
// y = a + b, dL/dA = dL/dY and dL/dB = dL/dY, so both
// paths contribute the upstream gradient.
func residualAddGrad(gradA, gradB *Tensor) *Tensor {
	if !sameShape(gradA.shape, gradB.shape) {
		panic(fmt.Sprintf("residualAddGrad: shape mismatch %v vs %v", gradA.shape, gradB.shape))
	}
	out := ZerosF32(gradA.shape...)
	aData, _ := gradA.ToF32()
	bData, _ := gradB.ToF32()
	for i := range aData {
		out.DataF32()[i] = aData[i] + bData[i]
	}
	if gradA.dtype != Float32 {
		alloc.Free(aData)
	}
	if gradB.dtype != Float32 {
		alloc.Free(bData)
	}
	return out
}

// allocFreeTensor frees a Tensor's underlying slice.
func allocFreeTensor(t *Tensor) {
	switch t.dtype {
	case Float32:
		alloc.Free(t.f32)
	case BFloat16:
		alloc.Free(t.bf16)
	case Float64:
		alloc.Free(t.f64)
	}
}
