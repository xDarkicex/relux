package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

// TestRMSNorm_HandComputed verifies RMSNorm.Forward against a
// hand-computed reference. For input [1, 2, 3, 4] with
// gamma=1, eps=0: meanSq = (1+4+9+16)/4 = 7.5; rstd =
// 1/sqrt(7.5) = 0.3651; output = [0.3651, 0.7303, 1.0954, 1.4606].
func TestRMSNorm_HandComputed(t *testing.T) {
	r := transformer.NewRMSNorm(4, 0)
	x := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	y := r.Forward(x)
	got := y.DataF32()
	want := []float32{0.3651, 0.7303, 1.0954, 1.4606}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 0.001 {
			t.Errorf("y[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestRMSNorm_IdentityAtInit(t *testing.T) {
	// With gamma=1 and eps=0, RMSNorm(x) = x / rms(x) where
	// rms(x) = sqrt(mean(x^2)). For x = [2, 2, 2, 2], rms=2,
	// output = [1, 1, 1, 1].
	r := transformer.NewRMSNorm(4, 0)
	x := transformer.NewTensor([]float32{2, 2, 2, 2}, 4)
	y := r.Forward(x)
	for i, v := range y.DataF32() {
		if math.Abs(float64(v-1.0)) > 0.001 {
			t.Errorf("y[%d] = %v, want 1.0", i, v)
		}
	}
}

func TestRMSNorm_BatchIndependent(t *testing.T) {
	// Two rows should be normalized independently.
	r := transformer.NewRMSNorm(4, 0)
	x := transformer.NewTensor([]float32{1, 2, 3, 4, 2, 2, 2, 2}, 2, 4)
	y := r.Forward(x)
	got := y.DataF32()
	// Row 0: rmsNorm as in TestRMSNorm_HandComputed.
	want0 := []float32{0.3651, 0.7303, 1.0954, 1.4606}
	for i, v := range got[:4] {
		if math.Abs(float64(v-want0[i])) > 0.001 {
			t.Errorf("row 0 [%d] = %v, want %v", i, v, want0[i])
		}
	}
	// Row 1: all 1.0.
	for i, v := range got[4:] {
		if math.Abs(float64(v-1.0)) > 0.001 {
			t.Errorf("row 1 [%d] = %v, want 1.0", i, v)
		}
	}
}

func TestRMSNorm_GammaScaling(t *testing.T) {
	// gamma = 2 should scale the output by 2.
	r := transformer.NewRMSNorm(4, 0)
	for i := range r.Params()[0].Data {
		r.Params()[0].Data[i] = transformer.BF16FromF32(2.0)
	}
	x := transformer.NewTensor([]float32{2, 2, 2, 2}, 4)
	y := r.Forward(x)
	for i, v := range y.DataF32() {
		if math.Abs(float64(v-2.0)) > 0.001 {
			t.Errorf("y[%d] = %v, want 2.0 (gamma=2)", i, v)
		}
	}
}

func TestRMSNorm_BackwardGradCheck(t *testing.T) {
	// Numerical gradient check on a small input. We perturb
	// each parameter by eps, recompute the loss (sum of
	// squared outputs), and verify the analytical gradient
	// matches the finite-difference approximation.
	const dModel = 4
	const eps = 0.05

	r := transformer.NewRMSNorm(dModel, 1e-5)
	xData := []float32{0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0}
	x := transformer.NewTensor(xData, 2, dModel)

	// Forward + backward.
	y := r.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	// gradOut = 2 * y
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, dModel)
	r.Backward(gradOut)

	// Numerical gradient for gamma.
	gammaParams := r.Params()[0]
	analytical := make([]float32, dModel)
	for i, v := range gammaParams.Grad {
		analytical[i] = float32(v)
	}
	for i := 0; i < dModel; i++ {
		orig := transformer.F32FromBF16(gammaParams.Data[i])
		gammaParams.Data[i] = transformer.BF16FromF32(orig + eps)
		yp := r.Forward(x)
		var lp float32
		for _, v := range yp.DataF32() {
			lp += v * v
		}
		gammaParams.Data[i] = transformer.BF16FromF32(orig - eps)
		ym := r.Forward(x)
		var lm float32
		for _, v := range ym.DataF32() {
			lm += v * v
		}
		gammaParams.Data[i] = transformer.BF16FromF32(orig)
		numerical := (lp - lm) / (2 * eps)
		// Restore the cache; Forward rewrote lastX.
		// (The cache is freed inside Backward; this Forward
		// rebuilt it.)
		// Compare. Tolerance is loose because bf16
		// quantization adds noise to both the analytical
		// gradient (computed in f32 from bf16 params) and
		// the numerical gradient (eps = 0.05 is small
		// relative to bf16's ~3% mantissa resolution near
		// these magnitudes).
		if math.Abs(float64(numerical-analytical[i])) > 5e-1 {
			t.Errorf("gamma.Grad[%d]: analytical=%v, numerical=%v", i, analytical[i], numerical)
		}
	}
}
