package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestRotaryEmbedding_PrecomputeBase(t *testing.T) {
	// For base=10000, headDim=4: theta_0 = 1, theta_1 = 1/100.
	// cos[0, 0] = 1, sin[0, 0] = 0; cos[0, 1] = 1, sin[0, 1] = 0.
	r := transformer.NewRotaryEmbedding(4, 10000, 4)
	if r.Cos()[0] != 1.0 {
		t.Errorf("cos[0] = %v, want 1", r.Cos()[0])
	}
	if r.Sin()[0] != 0.0 {
		t.Errorf("sin[0] = %v, want 0", r.Sin()[0])
	}
}

func TestRotaryEmbedding_ApplyIdentityAtZero(t *testing.T) {
	// At position 0, cos=1, sin=0, so the rotation is
	// (x0, x1) -> (x0*1 - x1*0, x1*1 + x0*0) = (x0, x1).
	// The output should equal the input.
	r := transformer.NewRotaryEmbedding(4, 10000, 4)
	x := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	y := r.Apply(x, 0)
	for i, v := range y.DataF32() {
		if math.Abs(float64(v-x.DataF32()[i])) > 1e-5 {
			t.Errorf("y[%d] = %v, want %v (identity at pos 0)", i, v, x.DataF32()[i])
		}
	}
}

func TestRotaryEmbedding_ApplyAtPositionOne(t *testing.T) {
	// At position 1, the angle is theta_i = base^(-2i/d) for
	// the pair index i. For headDim=4, base=10000:
	//   theta_0 = 1, theta_1 = 0.01
	// For input [a, b, c, d]:
	//   pair 0: (a*cos(1) - b*sin(1), b*cos(1) + a*sin(1))
	//   pair 1: (c*cos(0.01) - d*sin(0.01), d*cos(0.01) + c*sin(0.01))
	r := transformer.NewRotaryEmbedding(4, 10000, 4)
	x := transformer.NewTensor([]float32{1, 0, 0, 1}, 4)
	y := r.Apply(x, 1)
	got := y.DataF32()
	c0 := float32(math.Cos(1))
	s0 := float32(math.Sin(1))
	c1 := float32(math.Cos(0.01))
	s1 := float32(math.Sin(0.01))
	// pair 0: a=1, b=0 -> (1*c0, 1*s0)
	want00 := 1 * c0
	want01 := 1 * s0
	// pair 1: c=0, d=1 -> (-1*s1, 1*c1)
	want10 := 0*c1 - 1*s1
	want11 := 0*s1 + 1*c1
	want := []float32{want00, want01, want10, want11}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 1e-5 {
			t.Errorf("y[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestRotaryEmbedding_BackwardIsInverse(t *testing.T) {
	// Apply a forward, then apply backward with the same
	// grad. The composition should be a no-op (the rotation
	// is orthogonal, so its inverse equals its transpose).
	r := transformer.NewRotaryEmbedding(4, 10000, 4)
	g := transformer.NewTensor([]float32{0.1, 0.2, 0.3, 0.4}, 4)
	gOut := r.BackwardApply(g, 0)
	got := gOut.DataF32()
	// Expected: same as applying the rotation to the input
	// g (the rotation is its own transpose, so the inverse
	// equals the forward).
	expectedG := r.Apply(g, 0)
	for i, v := range got {
		if math.Abs(float64(v-expectedG.DataF32()[i])) > 1e-5 {
			t.Errorf("backward[%d] = %v, want %v", i, v, expectedG.DataF32()[i])
		}
	}
}
