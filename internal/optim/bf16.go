package optim

import "math"

// bf16ToFloat32 widens a bfloat16 (the bit pattern in the
// low 16 bits of the bf16 uint16) to a float32. This is
// the standard zero-extend pattern: the bf16 bits become
// the high 16 bits of the float32; the low 16 bits are
// zero.
func bf16ToFloat32(bf uint16) float32 {
	return math.Float32frombits(uint32(bf) << 16)
}

// float32ToBF16 narrows a float32 to bfloat16. The
// conversion is the standard "truncate the low 16 bits of
// the mantissa" pattern with round-to-nearest-even. The
// dynamic range (exponent) is preserved; the mantissa
// loses ~16 bits of precision.
func float32ToBF16(x float32) uint16 {
	u := math.Float32bits(x)
	// Round-to-nearest-even on the truncation point.
	roundingBias := uint32(0x7FFF) + ((u >> 16) & 1)
	u += roundingBias
	return uint16(u >> 16)
}

// bf16AddFloat32 reads a bf16 weight, adds a float32
// delta, and writes the result back as bf16. The math
// happens at float32 precision (so the addition is exact);
// the result is downcast to bf16 for storage.
//
// This is the hot path for the optimizer's step: every
// weight update is one such read-modify-write.
func bf16AddFloat32(w float32, delta float32) uint16 {
	return float32ToBF16(w + delta)
}
