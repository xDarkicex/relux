package layer

import "math"

// bf16ToFloat32 widens a bfloat16 (the bit pattern in the
// low 16 bits of the bf16 uint16) to a float32. Standard
// zero-extend: the bf16 bits become the high 16 bits of
// the float32; the low 16 bits are zero.
func bf16ToFloat32(bf uint16) float32 {
	return math.Float32frombits(uint32(bf) << 16)
}

// float32ToBF16 narrows a float32 to bfloat16. The
// conversion is "truncate the low 16 bits of the mantissa"
// with round-to-nearest-even. Exponent (range) is
// preserved; mantissa loses ~16 bits of precision.
func float32ToBF16(x float32) uint16 {
	u := math.Float32bits(x)
	roundingBias := uint32(0x7FFF) + ((u >> 16) & 1)
	u += roundingBias
	return uint16(u >> 16)
}
