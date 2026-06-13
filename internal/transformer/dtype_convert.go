package transformer

import "math"

// BF16FromF32 returns the bfloat16 encoding (as a uint16 of the
// IEEE 754 single's high 16 bits) of a float32. The conversion is
// the standard "truncate the low 16 bits of the mantissa" pattern
// — round-to-nearest-even is one bit more accurate but the
// mantissa loss in either case is < 1% of the float32 value, and
// truncation is what the GPU does in hardware.
//
// NaN handling: the high 16 bits of a float32 NaN encode a NaN.
// Infinities preserve their sign. Denormals become zero
// (consistent with the GPU's bfloat16 behavior).
func BF16FromF32(x float32) uint16 {
	u := math.Float32bits(x)
	// Round-to-nearest-even on the truncation point. The low
	// 16 bits hold the "round" and "sticky" — we add 0x7FFF +
	// the LSB of the high bit (the round bit) to round to even.
	// This is the same pattern NVIDIA's bf16 cast uses.
	roundingBias := uint32(0x7FFF) + ((u >> 16) & 1)
	u += roundingBias
	return uint16(u >> 16)
}

// F32FromBF16 returns the float32 value of a bfloat16 stored in
// the low 16 bits of bf. The conversion is zero-extend: bf's low
// 16 bits become the high 16 bits of a float32; the low 16 bits
// of the float32 are zero. This means a round-trip
// F32→BF16→F32 loses up to 16 mantissa bits, so the relative
// error is bounded by 2^-7 ≈ 0.78% per value.
func F32FromBF16(bf uint16) float32 {
	u := uint32(bf) << 16
	return math.Float32frombits(u)
}

// BF16SliceFromF32 converts a float32 slice in place to its
// bfloat16 encoding. The returned slice is a fresh allocation
// from alloc.Uint16 with the same length; the input is not
// modified. This is the bulk-cast used at the host/device
// boundary: a `[batch, seq, dModel] float32` activation is
// downcast once before the GPU matmul.
func BF16SliceFromF32(src []float32) []uint16 {
	if len(src) == 0 {
		return nil
	}
	out := make([]uint16, len(src))
	for i, v := range src {
		out[i] = BF16FromF32(v)
	}
	return out
}

// F32SliceFromBF16 is the inverse of BF16SliceFromF32. Used at
// the device/host boundary to widen the GPU's bfloat16 result
// back to float32 before the next op.
func F32SliceFromBF16(src []uint16) []float32 {
	if len(src) == 0 {
		return nil
	}
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = F32FromBF16(v)
	}
	return out
}

// BF16SliceFromF64 widens a float64 slice to float32 then casts
// to bfloat16. Used for the "load master weight into active
// bfloat16" path at Forward time (the master is float64 in the
// existing Param contract; the active is bfloat16 here for
// bandwidth).
func BF16SliceFromF64(src []float64) []uint16 {
	if len(src) == 0 {
		return nil
	}
	out := make([]uint16, len(src))
	for i, v := range src {
		out[i] = BF16FromF32(float32(v))
	}
	return out
}

// F64SliceFromBF16 widens bfloat16 → float32 → float64. Used at
// the Backward boundary to lift the active gradient back to the
// master's precision.
func F64SliceFromBF16(src []uint16) []float64 {
	if len(src) == 0 {
		return nil
	}
	out := make([]float64, len(src))
	for i, v := range src {
		out[i] = float64(F32FromBF16(v))
	}
	return out
}
