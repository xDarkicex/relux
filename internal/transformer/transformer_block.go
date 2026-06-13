package transformer

import (
	"fmt"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
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
// MHA's 4 projections + MLP's 4 + 2 RMSNorm gammas. Backward
// calls each sub-module's Backward in reverse order.
type Block struct {
	BaseModule
	normAttn *RMSNorm
	mha      *MHA
	normMlp  *RMSNorm
	mlp      *MLP

	// Forward cache for the residual stream (used by
	// Backward to compute the residual connection's
	// gradient). lastX is the block's input; lastH is
	// the residual stream after the MHA.
	lastX *Tensor
	lastH *Tensor
}

// NewBlock constructs a single transformer block. rope is
// shared across all blocks (one RoPE module per model). dFF
// is typically 4x dModel.
func NewBlock(dModel, numHeads, numKVHeads, dFF int, rope *RotaryEmbedding, causal bool) *Block {
	return &Block{
		normAttn: NewRMSNorm(dModel, 1e-5),
		mha:      NewMHA(dModel, numHeads, numKVHeads, rope, causal),
		normMlp:  NewRMSNorm(dModel, 1e-5),
		mlp:      NewMLP(dModel, dFF),
	}
}

// Forward computes the block output. Input shape [batch,
// seq, dModel] -> output shape [batch, seq, dModel].
func (b *Block) Forward(x *Tensor) *Tensor {
	if x.Rank() != 3 || x.shape[2] != 0 {
		// The dModel check is done inside the submodules.
		_ = 0
	}
	b.lastX = x.Clone() // for the residual

	normed1 := b.normAttn.Forward(x)
	attnOut := b.mha.Forward(normed1)
	b.lastH = x // residual stream: x + attnOut
	postAttn := residualAdd(x, attnOut)

	normed2 := b.normMlp.Forward(postAttn)
	mlpOut := b.mlp.Forward(normed2)
	out := residualAdd(postAttn, mlpOut)

	return out
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

	// Split the gradient through the second residual add.
	// The MLP input was `postAttn = x + attnOut`. The MLP
	// output is `out = postAttn + mlpOut`. dL/d(postAttn)
	// = gradOut (residual) + dL/d(postAttn) (through the
	// MLP path: normMlp backward applied to mlpInGrad).
	mlpInGrad := b.mlp.Backward(gradOut)
	normMlpInGrad := b.normMlp.Backward(mlpInGrad)
	postAttnGrad := residualAddGrad(gradOut, normMlpInGrad)
	// Now postAttnGrad = dL/d(postAttn).
	// The MHA's residual add was x + attnOut. dL/d(MHA_in)
	// = dL/d(postAttn) -> MHA.Backward.
	attnInGrad := b.mha.Backward(postAttnGrad)
	// attnInGrad = dL/d(normed1). dL/d(x) = residual +
	// normAttn backward of attnInGrad.
	normAttnInGrad := b.normAttn.Backward(attnInGrad)
	gradIn := residualAddGrad(postAttnGrad, normAttnInGrad)

	// Free the cache.
	if b.lastX != nil {
		allocFreeTensor(b.lastX)
		b.lastX = nil
	}
	if b.lastH != nil {
		allocFreeTensor(b.lastH)
		b.lastH = nil
	}

	return gradIn
}

// Params returns all trainable parameters: MHA's 4 + MLP's 4
// + 2 RMSNorm gammas.
func (b *Block) Params() []optim.Param {
	out := []optim.Param{}
	out = append(out, b.normAttn.Params()...)
	out = append(out, b.mha.Params()...)
	out = append(out, b.normMlp.Params()...)
	out = append(out, b.mlp.Params()...)
	return out
}

// NormAttn returns the pre-attention RMSNorm.
func (b *Block) NormAttn() *RMSNorm { return b.normAttn }

// BlockMHA returns the multi-head attention module.
func (b *Block) BlockMHA() *MHA { return b.mha }

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
