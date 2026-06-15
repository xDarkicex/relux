package loader

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/alloc"
)

// GGMLType identifies the on-disk encoding of a GGUF tensor.
type GGMLType uint32

const (
	GGMLTypeF32     GGMLType = 0
	GGMLTypeF16     GGMLType = 1
	GGMLTypeQ4_0    GGMLType = 2
	GGMLTypeQ4_1    GGMLType = 3
	GGMLTypeQ5_0    GGMLType = 6
	GGMLTypeQ5_1    GGMLType = 7
	GGMLTypeQ8_0    GGMLType = 8
	GGMLTypeQ8_1    GGMLType = 9
	GGMLTypeQ2_K    GGMLType = 10
	GGMLTypeQ3_K    GGMLType = 11
	GGMLTypeQ4_K    GGMLType = 12
	GGMLTypeQ5_K    GGMLType = 13
	GGMLTypeQ6_K    GGMLType = 14
	GGMLTypeQ8_K    GGMLType = 15
	GGMLTypeBF16    GGMLType = 29
)

// ggmlTypeTrait returns block size and bytes per block for a type.
type ggmlTypeTrait struct {
	BlockSize int  // elements per block
	TypeSize  int  // bytes per block
	Quantized bool // true if the format is quantized
}

func ggmlTrait(d GGMLType) ggmlTypeTrait {
	switch d {
	case GGMLTypeF32:
		return ggmlTypeTrait{BlockSize: 1, TypeSize: 4, Quantized: false}
	case GGMLTypeF16, GGMLTypeBF16:
		return ggmlTypeTrait{BlockSize: 1, TypeSize: 2, Quantized: false}
	case GGMLTypeQ4_0:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 18, Quantized: true}
	case GGMLTypeQ4_1:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 20, Quantized: true}
	case GGMLTypeQ5_0:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 22, Quantized: true}
	case GGMLTypeQ5_1:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 24, Quantized: true}
	case GGMLTypeQ8_0:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 34, Quantized: true}
	case GGMLTypeQ8_1:
		return ggmlTypeTrait{BlockSize: 32, TypeSize: 36, Quantized: true}
	case GGMLTypeQ2_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 84, Quantized: true}
	case GGMLTypeQ3_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 110, Quantized: true}
	case GGMLTypeQ4_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 144, Quantized: true}
	case GGMLTypeQ5_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 176, Quantized: true}
	case GGMLTypeQ6_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 210, Quantized: true}
	case GGMLTypeQ8_K:
		return ggmlTypeTrait{BlockSize: 256, TypeSize: 292, Quantized: true}
	default:
		return ggmlTypeTrait{}
	}
}

// RequiredBytes returns how many bytes a tensor of numElements
// occupies in the file.
func ggmlRequiredBytes(d GGMLType, numElements int) int {
	t := ggmlTrait(d)
	if !t.Quantized {
		return numElements * t.TypeSize
	}
	blocks := (numElements + t.BlockSize - 1) / t.BlockSize
	return blocks * t.TypeSize
}

// Dequantize converts a raw tensor byte buffer into float32. The
// caller provides the output buffer (typically alloc.Float32 to keep
// the data off-heap). Returns the number of elements written.
//
// Supported formats: F32 (passthrough), F16, BF16, Q4_0, Q4_1, Q5_0,
// Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.
func Dequantize(data []byte, dtype GGMLType, numElements int, out []float32) (int, error) {
	t := ggmlTrait(dtype)
	if t.TypeSize == 0 {
		return 0, fmt.Errorf("dequant: unsupported GGML type %d", dtype)
	}
	need := ggmlRequiredBytes(dtype, numElements)
	if len(data) < need {
		return 0, fmt.Errorf("dequant: type %d needs %d bytes, got %d", dtype, need, len(data))
	}
	if len(out) < numElements {
		return 0, fmt.Errorf("dequant: out buffer too small (need %d, have %d)", numElements, len(out))
	}

	if !t.Quantized {
		return dequantizeUnquantized(data, dtype, numElements, out)
	}
	return dequantizeQuantized(data, dtype, numElements, out)
}

func dequantizeUnquantized(data []byte, dtype GGMLType, n int, out []float32) (int, error) {
	switch dtype {
	case GGMLTypeF32:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
			out[i] = math.Float32frombits(bits)
		}
		return n, nil
	case GGMLTypeF16:
		for i := 0; i < n; i++ {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		}
		return n, nil
	case GGMLTypeBF16:
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(data[i*2 : i*2+2])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return n, nil
	default:
		return 0, fmt.Errorf("dequant: non-quantized type %d unhandled", dtype)
	}
}

func dequantizeQuantized(data []byte, dtype GGMLType, n int, out []float32) (int, error) {
	t := ggmlTrait(dtype)
	blocks := (n + t.BlockSize - 1) / t.BlockSize
	offset := 0
	written := 0
	for i := 0; i < blocks; i++ {
		block := data[offset : offset+t.TypeSize]
		elements := t.BlockSize
		if written+elements > n {
			elements = n - written
		}
		if err := dequantizeOneBlock(block, dtype, out[written:written+elements]); err != nil {
			return written, err
		}
		written += elements
		offset += t.TypeSize
	}
	return written, nil
}

func dequantizeOneBlock(block []byte, dtype GGMLType, out []float32) error {
	switch dtype {
	case GGMLTypeQ4_0:
		return dequantQ4_0(block, out)
	case GGMLTypeQ4_1:
		return dequantQ4_1(block, out)
	case GGMLTypeQ5_0:
		return dequantQ5_0(block, out)
	case GGMLTypeQ5_1:
		return dequantQ5_1(block, out)
	case GGMLTypeQ8_0:
		return dequantQ8_0(block, out)
	case GGMLTypeQ4_K:
		return dequantQ4_K(block, out)
	case GGMLTypeQ5_K:
		return dequantQ5_K(block, out)
	case GGMLTypeQ6_K:
		return dequantQ6_K(block, out)
	default:
		return fmt.Errorf("dequant: quantized type %d not yet implemented", dtype)
	}
}

// Float16ToFloat32 converts IEEE 754 half-precision to single.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32((h >> 10) & 0x1F)
	man := uint32(h & 0x3FF)

	if exp == 0 {
		if man == 0 {
			// ±zero
			return math.Float32frombits(sign)
		}
		// subnormal: normalize
		for man&0x400 == 0 {
			man <<= 1
			exp--
		}
		exp++
		man &= 0x3FF
	} else if exp == 31 {
		// Inf / NaN
		if man == 0 {
			return math.Float32frombits(sign | 0x7F800000)
		}
		return math.Float32frombits(sign | 0x7F800000 | (man << 13))
	}

	realExp := exp + (127 - 15)
	realMan := man << 13
	return math.Float32frombits(sign | (realExp << 23) | realMan)
}

// Q4_0: 18-byte block, 32 elements. Layout: [f16 scale | 16 bytes = 32 4-bit values]
func dequantQ4_0Impl(data []byte, out []float32) {
	scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	for i := 0; i < 16; i++ {
		b := data[2+i]
		v0 := int(b & 0x0F)
		v1 := int(b >> 4)
		out[2*i+0] = (float32(v0) - 8) * scale
		out[2*i+1] = (float32(v1) - 8) * scale
	}
}

// Q4_1: 20-byte block, 32 elements. [f16 scale | f16 min | 16 bytes nibbles]
func dequantQ4_1Impl(data []byte, out []float32) {
	scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	min := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))
	for i := 0; i < 16; i++ {
		b := data[4+i]
		v0 := float32(b & 0x0F)
		v1 := float32(b >> 4)
		out[2*i+0] = v0*scale + min
		out[2*i+1] = v1*scale + min
	}
}

// Q5_0: 22-byte block, 32 elements. [f16 scale | 4 bytes hi bits | 16 bytes lo nibbles]
func dequantQ5_0Impl(data []byte, out []float32) {
	scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	qh := binary.LittleEndian.Uint32(data[2:6])
	for i := 0; i < 16; i++ {
		b := data[6+i]
		lo0 := b & 0x0F
		lo1 := b >> 4
		hi0 := (qh >> i) & 1
		hi1 := (qh >> (i + 16)) & 1
		v0 := float32(int(lo0)|(int(hi0)<<4)) - 16
		v1 := float32(int(lo1)|(int(hi1)<<4)) - 16
		out[2*i+0] = v0 * scale
		out[2*i+1] = v1 * scale
	}
}

// Q5_1: 24-byte block, 32 elements. [f16 scale | f16 min | 4 bytes hi | 16 bytes lo]
func dequantQ5_1Impl(data []byte, out []float32) {
	scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	min := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))
	qh := binary.LittleEndian.Uint32(data[4:8])
	for i := 0; i < 16; i++ {
		b := data[8+i]
		lo0 := b & 0x0F
		lo1 := b >> 4
		hi0 := (qh >> i) & 1
		hi1 := (qh >> (i + 16)) & 1
		v0 := float32(int(lo0)|(int(hi0)<<4))*scale + min
		v1 := float32(int(lo1)|(int(hi1)<<4))*scale + min
		out[2*i+0] = v0
		out[2*i+1] = v1
	}
}

// Q8_0: 34-byte block, 32 elements. [f16 scale | 32 bytes int8]
func dequantQ8_0Impl(data []byte, out []float32) {
	scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	for i := 0; i < 32; i++ {
		out[i] = float32(int8(data[2+i])) * scale
	}
}

// Q4_K: 256-element super-block, 144 bytes.
// Layout: [f16 scale (d) | f16 scale (dmin) | 12 bytes scales packed | 128 bytes 4-bit qs]
// 8 sub-blocks of 32 elements each.
func dequantQ4_KImpl(data []byte, out []float32) {
	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))
	scalesAndMins := data[4:16]
	qs := data[16:144]

	// Unpack 6-bit scales and 6-bit mins, 8 sub-blocks.
	scales := alloc.Int8(8)
	mins := alloc.Int8(8)

	// First 4 sub-blocks use 4-bit packed format in bytes 0..5.
	// Next 4 use 6-bit packed format in bytes 0..11.
	for i := 0; i < 4; i++ {
		scales[i] = int8(scalesAndMins[i] & 0x3F)
		mins[i] = int8(scalesAndMins[i+4] & 0x3F)
	}
	for i := 0; i < 4; i++ {
		idx := 8 + i
		scales[idx] = int8((scalesAndMins[8+i] & 0x0F) | ((scalesAndMins[i] >> 6) << 4))
		mins[idx] = int8((scalesAndMins[8+i] >> 4) | ((scalesAndMins[4+i] >> 6) << 4))
	}

	// Sign-extend 6-bit values to int8 (if high bit set, subtract 64).
	for i := range scales {
		if scales[i]&0x20 != 0 {
			scales[i] -= 64
		}
		if mins[i]&0x20 != 0 {
			mins[i] -= 64
		}
	}

	// 8 sub-blocks of 32 elements. Each sub-block uses one scale + min.
	for sb := 0; sb < 8; sb++ {
		sc := float32(scales[sb]) * d
		mn := float32(mins[sb]) * dmin
		// First half (16 elements) uses low 4 bits, second half high 4 bits.
		for j := 0; j < 16; j++ {
			q := qs[sb*16+j] & 0x0F
			out[sb*32+j] = float32(q)*sc - mn
		}
		for j := 0; j < 16; j++ {
			q := qs[sb*16+j] >> 4
			out[sb*32+16+j] = float32(q)*sc - mn
		}
	}
}

// Q5_K: 256-element super-block, 176 bytes.
// Layout: [f16 d | f16 dmin | 12 bytes scales/mins | 32 bytes hi bits | 128 bytes lo nibbles]
func dequantQ5_KImpl(data []byte, out []float32) {
	d := Float16ToFloat32(binary.LittleEndian.Uint16(data[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(data[2:4]))
	scalesAndMins := data[4:16]
	qh := data[16:48]
	qs := data[48:176]

	scales := alloc.Int8(8)
	mins := alloc.Int8(8)
	for i := 0; i < 4; i++ {
		scales[i] = int8(scalesAndMins[i] & 0x3F)
		mins[i] = int8(scalesAndMins[i+4] & 0x3F)
	}
	for i := 0; i < 4; i++ {
		idx := 8 + i
		scales[idx] = int8((scalesAndMins[8+i] & 0x0F) | ((scalesAndMins[i] >> 6) << 4))
		mins[idx] = int8((scalesAndMins[8+i] >> 4) | ((scalesAndMins[4+i] >> 6) << 4))
	}
	for i := range scales {
		if scales[i]&0x20 != 0 {
			scales[i] -= 64
		}
		if mins[i]&0x20 != 0 {
			mins[i] -= 64
		}
	}

	for sb := 0; sb < 8; sb++ {
		sc := float32(scales[sb]) * d
		mn := float32(mins[sb]) * dmin
		// 5-bit value: 4 lo bits from qs, 1 hi bit from qh
		// qh is packed: bit at position (i) for value i.
		// 32 elements per sub-block split as 16 lo-half + 16 hi-half.
		for j := 0; j < 16; j++ {
			idx := sb*32 + j
			lo := qs[sb*16+j] & 0x0F
			hi := (qh[idx/8] >> (idx % 8)) & 1
			v := float32(int(lo)|(int(hi)<<4)) - 16
			out[idx] = v*sc - mn
		}
		for j := 0; j < 16; j++ {
			idx := sb*32 + 16 + j
			lo := qs[sb*16+j] >> 4
			hi := (qh[idx/8] >> (idx % 8)) & 1
			v := float32(int(lo)|(int(hi)<<4)) - 16
			out[idx] = v*sc - mn
		}
	}
}

// Q6_K: 256-element super-block, 210 bytes (last 2 bytes padding).
// Layout: [128 bytes lo nibbles (ql) | 64 bytes hi 2 bits (qh) | 16 bytes scales | 2 bytes pad]
// 16 sub-blocks of 16 elements each.
func dequantQ6_KImpl(data []byte, out []float32) {
	ql := data[0:128]
	qh := data[128:192]
	sc := data[192:208]
	// data[208:210] is padding

	for sb := 0; sb < 16; sb++ {
		scale := float32(int8(sc[sb]))
		// 16 elements per sub-block.
		// qh is 4 bits per element (only 2 used, but layout uses 4 bits).
		// Actually qh has 2 bits per element, packed 4 per byte.
		for j := 0; j < 16; j++ {
			idx := sb*16 + j
			lo := ql[idx] & 0x0F
			hi := (qh[idx/2] >> uint((idx%2)*4)) & 0x03
			v := float32(int(lo)|(int(hi)<<4)) - 32
			out[idx] = v * scale
		}
	}
}

// Wrappers with consistent (data, out) signature for the dispatch table.
func dequantQ4_0(data []byte, out []float32) error { dequantQ4_0Impl(data, out); return nil }
func dequantQ4_1(data []byte, out []float32) error { dequantQ4_1Impl(data, out); return nil }
func dequantQ5_0(data []byte, out []float32) error { dequantQ5_0Impl(data, out); return nil }
func dequantQ5_1(data []byte, out []float32) error { dequantQ5_1Impl(data, out); return nil }
func dequantQ8_0(data []byte, out []float32) error { dequantQ8_0Impl(data, out); return nil }
func dequantQ4_K(data []byte, out []float32) error  { dequantQ4_KImpl(data, out); return nil }
func dequantQ5_K(data []byte, out []float32) error  { dequantQ5_KImpl(data, out); return nil }
func dequantQ6_K(data []byte, out []float32) error  { dequantQ6_KImpl(data, out); return nil }
