package transformer_test

import (
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestMLP_ForwardShape(t *testing.T) {
	m := transformer.NewMLP(4, 8, transformer.FFNGELU)
	x := transformer.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 1, 4)
	y := m.Forward(x)
	if y.Rank() != 3 {
		t.Errorf("y.Rank() = %d, want 3", y.Rank())
	}
	if y.Shape()[0] != 2 || y.Shape()[1] != 1 || y.Shape()[2] != 4 {
		t.Errorf("y shape = %v, want [2 1 4]", y.Shape())
	}
}

func TestMLP_ParamCount(t *testing.T) {
	m := transformer.NewMLP(4, 8, transformer.FFNGELU)
	// W1 is 4*8=32, b1 is 8, W2 is 8*4=32, b2 is 4. Total 76.
	want := 32 + 8 + 32 + 4
	if got := len(m.Params()[0].Data) + len(m.Params()[1].Data) + len(m.Params()[2].Data) + len(m.Params()[3].Data); got != want {
		t.Errorf("total params = %d, want %d", got, want)
	}
}

func TestMLP_GELUExactValues(t *testing.T) {
	// Reference: gelu(0) = 0, gelu(1) = 0.5 * 1 * (1 + erf(1/sqrt(2)))
	//           = 0.5 * (1 + 0.6827...) = 0.8413
	// gelu(-1) = -0.5 * (1 - 0.6827) = -0.1587
	// We don't expose the helper directly; this is checked
	// indirectly via the forward pass with hand-picked inputs.
	m := transformer.NewMLP(1, 1, transformer.FFNGELU)
	// W1 = I_1, b1 = 0, W2 = I_1, b2 = 0. The MLP becomes
	// gelu(x) -> y = gelu(x). Set by overriding the params.
	for i := range m.Params()[0].Data {
		m.Params()[0].Data[i] = transformer.BF16FromF32(1.0) // W1[0,0] = 1
	}
	for i := range m.Params()[2].Data {
		m.Params()[2].Data[i] = transformer.BF16FromF32(1.0) // W2[0,0] = 1
	}
	x := transformer.NewTensor([]float32{1.0}, 1, 1, 1)
	y := m.Forward(x)
	got := y.DataF32()[0]
	want := float32(0.8413)
	if got < want-0.01 || got > want+0.01 {
		t.Errorf("gelu(1) = %v, want ~%v", got, want)
	}
}

func TestMLP_BackwardPanicsInV1(t *testing.T) {
	// MLP.Backward now works — the test below verifies it.
	// The original "panics in v1" test is replaced with
	// grad-check tests.
}

func TestMLP_BackwardGradCheck(t *testing.T) {
	// Numerical gradient check on W1[0, 0]. We perturb
	// the parameter, recompute the loss, and verify the
	// analytical gradient matches the finite-difference
	// approximation.
	const dModel, dFF = 2, 4
	const eps = 0.05

	m := transformer.NewMLP(dModel, dFF, transformer.FFNGELU)
	xData := []float32{0.5, 1.0, -0.5, 1.0, 0.3, 0.7, -0.3, 0.7}
	x := transformer.NewTensor(xData, 2, 2, dModel)
	m.SetMode(transformer.Train)

	y := m.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, 2, dModel)
	m.Backward(gradOut)

	// Numerical gradient for W1[0, 0].
	w1 := m.Params()[0]
	analytical := float32(w1.Grad[0])
	orig := transformer.F32FromBF16(w1.Data[0])
	w1.Data[0] = transformer.BF16FromF32(orig + eps)
	yp := m.Forward(x)
	var lp float32
	for _, v := range yp.DataF32() {
		lp += v * v
	}
	w1.Data[0] = transformer.BF16FromF32(orig - eps)
	ym := m.Forward(x)
	var lm float32
	for _, v := range ym.DataF32() {
		lm += v * v
	}
	w1.Data[0] = transformer.BF16FromF32(orig)
	numerical := (lp - lm) / (2 * eps)
	if absDiff(numerical, analytical) > 1e-2 {
		t.Errorf("W1.Grad[0,0]: analytical=%v, numerical=%v", analytical, numerical)
	}
}

func TestMLP_BackwardGradCheck_Bias(t *testing.T) {
	// Numerical gradient check on b1[0].
	const dModel, dFF = 2, 4
	const eps = 0.05

	m := transformer.NewMLP(dModel, dFF, transformer.FFNGELU)
	xData := []float32{0.5, 1.0, -0.5, 1.0}
	x := transformer.NewTensor(xData, 1, 2, dModel)
	m.SetMode(transformer.Train)

	y := m.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 1, 2, dModel)
	m.Backward(gradOut)

	b1 := m.Params()[1]
	analytical := float32(b1.Grad[0])
	orig := transformer.F32FromBF16(b1.Data[0])
	b1.Data[0] = transformer.BF16FromF32(orig + eps)
	yp := m.Forward(x)
	var lp float32
	for _, v := range yp.DataF32() {
		lp += v * v
	}
	b1.Data[0] = transformer.BF16FromF32(orig - eps)
	ym := m.Forward(x)
	var lm float32
	for _, v := range ym.DataF32() {
		lm += v * v
	}
	b1.Data[0] = transformer.BF16FromF32(orig)
	numerical := (lp - lm) / (2 * eps)
	if absDiff(numerical, analytical) > 1e-2 {
		t.Errorf("b1.Grad[0]: analytical=%v, numerical=%v", analytical, numerical)
	}
}

// absDiff is defined in mha_test.go.
