package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

// TestLayerNorm_HandComputed verifies LayerNorm.Forward
// against a hand-computed reference. For input [1, 2, 3, 4]
// with gamma=1, beta=0: mean=2.5, var=((1-2.5)^2 + ...)/4
// = 1.25, std=sqrt(1.25)≈1.1180, output = [(1-2.5)/1.1180,
// (2-2.5)/1.1180, ..., (4-2.5)/1.1180] = [-1.342, -0.447,
// 0.447, 1.342].
func TestLayerNorm_HandComputed(t *testing.T) {
	l := transformer.NewLayerNorm(4, 0)
	x := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	y := l.Forward(x)
	got := y.DataF32()
	want := []float32{-1.342, -0.447, 0.447, 1.342}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 0.01 {
			t.Errorf("y[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestLayerNorm_IdentityAtInit(t *testing.T) {
	// With gamma=1, beta=0, the output is the centered-and-
	// rescaled version of x — not the identity. This is
	// different from RMSNorm because LayerNorm subtracts the
	// mean. The test verifies the centered-and-scaled value
	// for a constant input: constant -> zero output.
	l := transformer.NewLayerNorm(4, 0)
	x := transformer.NewTensor([]float32{5, 5, 5, 5}, 4)
	y := l.Forward(x)
	for i, v := range y.DataF32() {
		if math.Abs(float64(v)) > 1e-5 {
			t.Errorf("y[%d] = %v, want 0 (constant input is zero after LayerNorm)", i, v)
		}
	}
}

func TestLayerNorm_BetaShifts(t *testing.T) {
	// beta=1 shifts the output by 1 (relative to gamma=1,
	// beta=0).
	l := transformer.NewLayerNorm(4, 0)
	for i := range l.Params()[1].Data {
		l.Params()[1].Data[i] = transformer.BF16FromF32(1.0)
	}
	x := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	y := l.Forward(x)
	// y[i] = norm[i] + 1 where norm is from
	// TestLayerNorm_HandComputed.
	want := []float32{-0.342, 0.553, 1.447, 2.342}
	for i, v := range y.DataF32() {
		if math.Abs(float64(v-want[i])) > 0.01 {
			t.Errorf("y[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestLayerNorm_GradCheck(t *testing.T) {
	// Numerical gradient check for gamma.
	const dModel = 4
	const eps = 0.05

	l := transformer.NewLayerNorm(dModel, 1e-5)
	xData := []float32{0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0}
	x := transformer.NewTensor(xData, 2, dModel)

	y := l.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, dModel)
	l.Backward(gradOut)

	gamma := l.Params()[0]
	analytical := make([]float32, dModel)
	for i, v := range gamma.Grad {
		analytical[i] = v
	}
	for i := 0; i < dModel; i++ {
		orig := transformer.F32FromBF16(gamma.Data[i])
		gamma.Data[i] = transformer.BF16FromF32(orig + eps)
		yp := l.Forward(x)
		var lp float32
		for _, v := range yp.DataF32() {
			lp += v * v
		}
		gamma.Data[i] = transformer.BF16FromF32(orig - eps)
		ym := l.Forward(x)
		var lm float32
		for _, v := range ym.DataF32() {
			lm += v * v
		}
		gamma.Data[i] = transformer.BF16FromF32(orig)
		numerical := (lp - lm) / (2 * eps)
		if math.Abs(float64(numerical-analytical[i])) > 5e-1 {
			t.Errorf("gamma.Grad[%d]: analytical=%v, numerical=%v", i, analytical[i], numerical)
		}
	}
}
