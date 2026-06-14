package transformer_test

import (
	"testing"

	"github.com/xDarkicex/relux/internal/optim"
	"github.com/xDarkicex/relux/internal/transformer"
)

func TestBlock_ForwardShape(t *testing.T) {
	b := transformer.NewBlock(4, 2, 1, 8, nil, true, false)
	x := transformer.NewTensor([]float32{1, 0, 0, 1, 0, 1, 1, 0}, 2, 1, 4)
	y := b.Forward(x)
	if y.Rank() != 3 {
		t.Errorf("y.Rank() = %d, want 3", y.Rank())
	}
	if y.Shape()[0] != 2 || y.Shape()[1] != 1 || y.Shape()[2] != 4 {
		t.Errorf("y shape = %v, want [2 1 4]", y.Shape())
	}
}

func TestBlock_ParamCount(t *testing.T) {
	// Block has 2 RMSNorms (1 param each) + MHA (4) + MLP (4) = 10 params.
	b := transformer.NewBlock(4, 2, 1, 8, nil, true, false)
	if got := len(b.Params()); got != 10 {
		t.Errorf("block Params count = %d, want 10", got)
	}
}

func TestBlock_BackwardGradCheck_Wq(t *testing.T) {
	// Numerical gradient check on MHA.Wq[0, 0] via the
	// Block composition. The norm's first gamma can be
	// near-identity (random init produces a near-identity
	// first-norm output, so dgamma is small); MHA.Wq has
	// a strong effect on the output.
	const dModel, numHeads, numKVHeads, dFF = 4, 2, 1, 8
	const eps = 0.05

	b := transformer.NewBlock(dModel, numHeads, numKVHeads, dFF, nil, true, false)
	xData := []float32{0.5, 1.0, -0.5, 1.0, 0.3, 0.7, -0.3, 0.7}
	x := transformer.NewTensor(xData, 2, 1, dModel)
	b.SetMode(transformer.Train)

	y := b.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, 1, dModel)
	b.Backward(gradOut)

	// Find Wq in the Block's params. Param order:
	// normAttn.gamma, mha.Wq, mha.Wk, mha.Wv, mha.Wo, normMlp.gamma,
	// mlp.W1, mlp.b1, mlp.W2, mlp.b2.
	params := b.Params()
	var wq *optim.Param
	for _, p := range params {
		if p.Name == "mha.Wq" {
			wq = &p
			break
		}
	}
	if wq == nil {
		t.Fatal("could not find mha.Wq in Block.Params()")
	}
	analytical := float32(wq.Grad[0])
	orig := transformer.F32FromBF16(wq.Data[0])
	wq.Data[0] = transformer.BF16FromF32(orig + eps)
	yp := b.Forward(x)
	var lp float32
	for _, v := range yp.DataF32() {
		lp += v * v
	}
	wq.Data[0] = transformer.BF16FromF32(orig - eps)
	ym := b.Forward(x)
	var lm float32
	for _, v := range ym.DataF32() {
		lm += v * v
	}
	wq.Data[0] = transformer.BF16FromF32(orig)
	numerical := (lp - lm) / (2 * eps)
	if absDiff(numerical, analytical) > 5e-2 {
		t.Errorf("mha.Wq.Grad[0]: analytical=%v, numerical=%v", analytical, numerical)
	}
}

func TestBlock_BackwardGradCheck_MlpW1(t *testing.T) {
	// Numerical gradient check on MLP.W1[0, 0] via the Block.
	const dModel, numHeads, numKVHeads, dFF = 4, 2, 1, 8
	const eps = 0.05

	b := transformer.NewBlock(dModel, numHeads, numKVHeads, dFF, nil, true, false)
	xData := []float32{0.5, 1.0, -0.5, 1.0, 0.3, 0.7, -0.3, 0.7}
	x := transformer.NewTensor(xData, 2, 1, dModel)
	b.SetMode(transformer.Train)

	y := b.Forward(x)
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, 1, dModel)
	b.Backward(gradOut)

	params := b.Params()
	var w1 *optim.Param
	for _, p := range params {
		if p.Name == "mlp.W1" {
			w1 = &p
			break
		}
	}
	if w1 == nil {
		t.Fatal("could not find mlp.W1 in Block.Params()")
	}
	analytical := float32(w1.Grad[0])
	orig := transformer.F32FromBF16(w1.Data[0])
	w1.Data[0] = transformer.BF16FromF32(orig + eps)
	yp := b.Forward(x)
	var lp float32
	for _, v := range yp.DataF32() {
		lp += v * v
	}
	w1.Data[0] = transformer.BF16FromF32(orig - eps)
	ym := b.Forward(x)
	var lm float32
	for _, v := range ym.DataF32() {
		lm += v * v
	}
	w1.Data[0] = transformer.BF16FromF32(orig)
	numerical := (lp - lm) / (2 * eps)
	if absDiff(numerical, analytical) > 5e-2 {
		t.Errorf("mlp.W1.Grad[0]: analytical=%v, numerical=%v", analytical, numerical)
	}
}

func TestBlock_GradientCheckpointing(t *testing.T) {
	dModel := 8
	numHeads := 4
	numKVHeads := 2
	dFF := 16

	// Two identical blocks: one with checkpointing, one without.
	bNoCkpt := transformer.NewBlock(dModel, numHeads, numKVHeads, dFF, nil, true, false)
	bCkpt := transformer.NewBlock(dModel, numHeads, numKVHeads, dFF, nil, true, true)

	// Copy weights.
	for i, p := range bNoCkpt.Params() {
		copy(bCkpt.Params()[i].Data, p.Data)
	}

	// Same input.
	xData := make([]float32, 1*1*dModel)
	for i := range xData {
		xData[i] = float32(i+1) / float32(dModel)
	}
	x := transformer.NewTensor(xData, 1, 1, dModel)

	// Forward both.
	yNo := bNoCkpt.Forward(x.Clone())
	yCkpt := bCkpt.Forward(x.Clone())

	// Outputs must match.
	for i, v := range yNo.DataF32() {
		if v != yCkpt.DataF32()[i] {
			t.Errorf("forward output[%d] mismatch: noCkpt=%v ckpt=%v", i, v, yCkpt.DataF32()[i])
			break
		}
	}

	// Backward with same gradOut.
	gradOutData := make([]float32, len(yNo.DataF32()))
	for i, v := range yNo.DataF32() {
		gradOutData[i] = 2 * v
	}
	bNoCkpt.Backward(transformer.NewTensor(gradOutData, 1, 1, dModel))
	bCkpt.Backward(transformer.NewTensor(gradOutData, 1, 1, dModel))

	// Gradients must match.
	for i, pNo := range bNoCkpt.Params() {
		pCkpt := bCkpt.Params()[i]
		for j := range pNo.Grad {
			if pNo.Grad[j] != pCkpt.Grad[j] {
				t.Errorf("gradient mismatch param %q[%d]: noCkpt=%v ckpt=%v",
					pNo.Name, j, pNo.Grad[j], pCkpt.Grad[j])
				break
			}
		}
	}
}
