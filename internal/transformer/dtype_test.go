package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestBF16RoundTrip_Typical(t *testing.T) {
	// A few representative values: a normal positive, a small
	// fraction, a normal negative, a value near zero, and a
	// large value. The relative error after F32→BF16→F32 is
	// bounded by 2^-7 ≈ 0.0078125 (the mantissa truncation).
	cases := []float32{1.0, -1.0, 0.5, -0.5, 0.001, 1e-6, 1234.5678, -9876.543}
	for _, x := range cases {
		bf := transformer.BF16FromF32(x)
		got := transformer.F32FromBF16(bf)
		// The round-trip recovers x exactly when x is already
		// representable in 16 bits (i.e. when its float32
		// mantissa has at most 7 significant bits). For
		// arbitrary x, the relative error is bounded by 2^-7
		// of the absolute magnitude of x.
		if x == 0 {
			if got != 0 {
				t.Errorf("F32FromBF16(BF16FromF32(0)) = %v, want 0", got)
			}
			continue
		}
		relErr := math.Abs(float64(got-x)) / math.Abs(float64(x))
		if relErr > 0.01 {
			t.Errorf("BF16 round-trip relative error too large for %v: got %v, relErr %v", x, got, relErr)
		}
	}
}

func TestBF16RoundTrip_ExactForPowersOfTwo(t *testing.T) {
	// Powers of two are exactly representable in bfloat16's
	// 7-bit mantissa because the mantissa is just the integer
	// power plus a leading 1 at the right position.
	for p := -20.0; p <= 20.0; p++ {
		x := float32(math.Pow(2, p))
		bf := transformer.BF16FromF32(x)
		got := transformer.F32FromBF16(bf)
		if got != x {
			t.Errorf("2^%v: round-trip = %v, want %v", p, got, x)
		}
	}
}

func TestBF16RoundTrip_SpecialValues(t *testing.T) {
	// +Inf, -Inf, +0, -0. The special values should be
	// preserved (NaN is not tested; it can lose the quiet-bit
	// signal under truncation, which is acceptable).
	if got := transformer.F32FromBF16(transformer.BF16FromF32(float32(math.Inf(1)))); !math.IsInf(float64(got), 1) {
		t.Errorf("+Inf round-trip = %v, want +Inf", got)
	}
	if got := transformer.F32FromBF16(transformer.BF16FromF32(float32(math.Inf(-1)))); !math.IsInf(float64(got), -1) {
		t.Errorf("-Inf round-trip = %v, want -Inf", got)
	}
	if got := transformer.F32FromBF16(transformer.BF16FromF32(0)); got != 0 {
		t.Errorf("0 round-trip = %v, want 0", got)
	}
	if got := transformer.F32FromBF16(transformer.BF16FromF32(float32(math.Copysign(0, -1)))); !math.Signbit(float64(got)) || got != 0 {
		t.Errorf("-0 round-trip = %v, want -0", got)
	}
}

func TestBF16SliceRoundTrip(t *testing.T) {
	src := []float32{0.1, 0.2, 0.3, 1.5, -2.5, 100.25}
	bf := transformer.BF16SliceFromF32(src)
	if len(bf) != len(src) {
		t.Fatalf("bf len = %d, want %d", len(bf), len(src))
	}
	got := transformer.F32SliceFromBF16(bf)
	for i := range src {
		if math.Abs(float64(got[i]-src[i])) > 0.01*math.Abs(float64(src[i])) {
			t.Errorf("bf16 slice[%d]: got %v, want %v", i, got[i], src[i])
		}
	}
}

func TestBF16SliceFromF64(t *testing.T) {
	// The float64 → float32 → bfloat16 path. The 1e-40 value
	// underflows when narrowed to float32 (becomes 0); the
	// 1.0 should round-trip exactly.
	src := []float64{1.0, -1.0, 0.5, 100.5}
	bf := transformer.BF16SliceFromF64(src)
	if len(bf) != len(src) {
		t.Fatalf("bf len = %d, want %d", len(bf), len(src))
	}
	for i, v := range src {
		got := transformer.F32FromBF16(bf[i])
		want := float32(v)
		if got != want {
			t.Errorf("f64→bf16→f32 [%d]: got %v, want %v (src %v)", i, got, want, v)
		}
	}
}

func TestDTypeString(t *testing.T) {
	cases := map[transformer.DType]string{
		transformer.Float32: "f32",
		transformer.BFloat16: "bf16",
		transformer.Float64: "f64",
	}
	for d, want := range cases {
		if d.String() != want {
			t.Errorf("DType(%d).String() = %q, want %q", d, d.String(), want)
		}
	}
}

func TestDTypeBytesPerElem(t *testing.T) {
	cases := map[transformer.DType]int{
		transformer.Float32: 4,
		transformer.BFloat16: 2,
		transformer.Float64: 8,
	}
	for d, want := range cases {
		if d.BytesPerElem() != want {
			t.Errorf("DType(%d).BytesPerElem() = %d, want %d", d, d.BytesPerElem(), want)
		}
	}
}
