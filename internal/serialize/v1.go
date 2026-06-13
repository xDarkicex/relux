// Package serialize contains the v1 binary format for .relux
// files. See v1_read.go for the read path.
//
// The format is little-endian, fixed-width integers. Weights
// and Adam optimizer state are stored as bfloat16 (the
// industry standard for transformer weights — 2 bytes per
// value, 8 bits of exponent matching float32's dynamic
// range, 7 bits of mantissa). The arch block stores dims
// as u32 and hyperparameters (e.g. RMSNorm eps) as f32 —
// those are tiny and don't justify bf16.
//
// The writer is a single forward pass:
//
//	- The header's first 20 bytes (magic + version + flags +
//	  num_layers + total_params) are written directly to the
//	  underlying writer. They are NOT covered by the body
//	  SHA-256. They ARE covered by the header CRC32 at offset
//	  20-23. The remaining 8 bytes of the 32-byte header are
//	  reserved zeros.
//	- Everything after the 32-byte header (arch block + weight
//	  block + optimizer state block) is the "body". Body bytes
//	  are written to the underlying writer AND fed to a SHA-256
//	  hasher as they are written.
//	- The 32-byte footer is the SHA-256 of the body. The writer
//	  writes it at the end via WriteFooter.
//
// This split lets the reader verify both:
//  1. The header is intact (CRC32 mismatch = corruption or
//     wrong file).
//  2. The body is intact (SHA-256 mismatch = bit-flip or
//     tampering anywhere in the arch/weights/optim state).
package serialize

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"

	"github.com/xDarkicex/relux/internal/optim"
)

// V1Magic is the four-byte "RELV" magic that identifies
// the v1 binary format. Use it from persist.go's Load
// path to sniff the format before dispatching.
var V1Magic = [4]byte{'R', 'E', 'L', 'V'}

// V1Version is the schema version implemented by this file.
const V1Version uint16 = 1

// V1HeaderSize is the on-disk size of the v1 header.
const V1HeaderSize = 32

// V1FooterSize is the on-disk size of the v1 footer (the
// SHA-256 of the body).
const V1FooterSize = 32

// Layer tags. The arch block is a flat list of entries;
// each entry starts with one of these tag bytes.
const (
	LayerTagEmbedding uint8 = 1
	LayerTagRMSNorm   uint8 = 2
	LayerTagMHA       uint8 = 3
	LayerTagMLP       uint8 = 4
	LayerTagRoPE      uint8 = 5
	LayerTagLinear    uint8 = 6
)

// AdamState is the per-param Adam state we serialize in
// the optimizer state block. The reader reconstructs the
// full optim.State from these. M and V are float32 on
// disk — bf16 truncation of the optimizer's running
// averages (which are exponentially decaying sums of tiny
// gradients with beta2=0.999) destroys the precision
// needed for the variance estimate and causes
// quantization shocks on resume.
type AdamState struct {
	Name string
	M    []float32
	V    []float32
}

// V1Writer writes a .relux v1 file. See the package comment
// for the format overview.
//
// The writer is a single forward pass — no seeking. The
// caller (Transformer.Save) is responsible for the order
// of layer entries in the arch and the order of params in
// the weight block.
type V1Writer struct {
	w           io.Writer
	crc32       hash.Hash32
	bodyHasher  hash.Hash
	numLayers   int
	totalParams int64
	wroteHeader bool
	wroteFooter bool
}

// NewV1Writer constructs a writer. The caller is expected
// to call WriteHeader, then a sequence of arch entries
// and weights, then WriteOptimizerState, then WriteFooter.
func NewV1Writer(w io.Writer, numLayers int, totalParams int64) *V1Writer {
	return &V1Writer{
		w:           w,
		crc32:       crc32.NewIEEE(),
		bodyHasher:  sha256.New(),
		numLayers:   numLayers,
		totalParams: totalParams,
	}
}

// WriteHeader writes the 32-byte v1 header. Must be called
// before any other Write* call.
//
// Layout (little-endian):
//
//	[0..4)   magic        = "RELV"
//	[4..6)   version      = 1
//	[6..8)   flags        = 0 (reserved)
//	[8..12)  num_layers   = N
//	[12..20) total_params = P
//	[20..24) header_crc32 = CRC32 of [0..20)
//	[24..32) reserved     = 0 (8 zero bytes)
func (wr *V1Writer) WriteHeader() error {
	if wr.wroteHeader {
		return fmt.Errorf("serialize: WriteHeader called twice")
	}
	var pre [20]byte
	binary.LittleEndian.PutUint32(pre[0:4], binary.LittleEndian.Uint32(V1Magic[:]))
	binary.LittleEndian.PutUint16(pre[4:6], V1Version)
	binary.LittleEndian.PutUint16(pre[6:8], 0) // flags
	binary.LittleEndian.PutUint32(pre[8:12], uint32(wr.numLayers))
	binary.LittleEndian.PutUint64(pre[12:20], uint64(wr.totalParams))
	crc := crc32.ChecksumIEEE(pre[:])
	var full [32]byte
	copy(full[0:20], pre[:])
	binary.LittleEndian.PutUint32(full[20:24], crc)
	if _, err := wr.w.Write(full[:]); err != nil {
		return err
	}
	wr.wroteHeader = true
	return nil
}

// WriteArchEntry writes one arch entry: a tag byte and the
// typed payload for that layer. The reader knows the
// payload layout per tag.
func (wr *V1Writer) WriteArchEntry(tag uint8, dims []uint32, floats []float32) error {
	if !wr.wroteHeader {
		return fmt.Errorf("serialize: WriteArchEntry before WriteHeader")
	}
	if err := wr.writeU8(tag); err != nil {
		return err
	}
	for _, d := range dims {
		if err := wr.writeU32(d); err != nil {
			return err
		}
	}
	for _, f := range floats {
		if err := wr.writeF32(f); err != nil {
			return err
		}
	}
	return nil
}

// WriteWeight writes one weight: a u32 element count and
// the bfloat16 data (length elements, 2 bytes each,
// little-endian). The caller is responsible for matching
// the order of writes to the arch's layer order.
//
// Weights are stored as bfloat16 in the v1 file. The
// in-memory master is also bfloat16 (per the optim.Param
// contract), so this is a direct copy.
func (wr *V1Writer) WriteWeight(data []uint16) error {
	if !wr.wroteHeader {
		return fmt.Errorf("serialize: WriteWeight before WriteHeader")
	}
	if err := wr.writeU32(uint32(len(data))); err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	return wr.writeBF16s(data)
}

// WriteWeightF32 writes one weight as float32 (4 bytes
// per element, little-endian). Used for the Adam
// optimizer state block (m and v are float32 in memory
// and on disk — bf16 truncation destroys the precision
// needed for the variance estimate).
func (wr *V1Writer) WriteWeightF32(data []float32) error {
	if !wr.wroteHeader {
		return fmt.Errorf("serialize: WriteWeightF32 before WriteHeader")
	}
	if err := wr.writeU32(uint32(len(data))); err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	return wr.writeF32s(data)
}

// WriteOptimizerState writes the optimizer state block. v1
// supports Adam only; the kind string is required.
//
// m and v slices must be bfloat16. The kind is the
// optimizer identifier (e.g. "adam"); step is the
// optimizer's internal timestep.
func (wr *V1Writer) WriteOptimizerState(kind string, step uint64, states []AdamState) error {
	if !wr.wroteHeader {
		return fmt.Errorf("serialize: WriteOptimizerState before WriteHeader")
	}
	if err := wr.writeString(kind); err != nil {
		return err
	}
	if err := wr.writeU64(step); err != nil {
		return err
	}
	if err := wr.writeU32(uint32(len(states))); err != nil {
		return err
	}
	for _, s := range states {
		if err := wr.writeString(s.Name); err != nil {
			return err
		}
		if err := wr.WriteWeightF32(s.M); err != nil {
			return err
		}
		if err := wr.WriteWeightF32(s.V); err != nil {
			return err
		}
	}
	return nil
}

// WriteFooter writes the SHA-256 of the body. Must be the
// last call. Returns the SHA-256 sum (32 bytes) for
// callers that want to verify it.
func (wr *V1Writer) WriteFooter() ([32]byte, error) {
	if !wr.wroteHeader {
		return [32]byte{}, fmt.Errorf("serialize: WriteFooter before WriteHeader")
	}
	if wr.wroteFooter {
		return [32]byte{}, fmt.Errorf("serialize: WriteFooter called twice")
	}
	wr.wroteFooter = true
	sum := wr.bodyHasher.Sum(nil)
	if _, err := wr.w.Write(sum); err != nil {
		return [32]byte{}, err
	}
	var out [32]byte
	copy(out[:], sum)
	return out, nil
}

// writeU8 writes a single byte to the body (and the body hasher).
func (wr *V1Writer) writeU8(v uint8) error {
	return wr.writeBodyByte(v)
}

// writeU16 writes a little-endian u16 to the body.
func (wr *V1Writer) writeU16(v uint16) error {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	return wr.writeBodyBytes(b[:])
}

// writeU32 writes a little-endian u32 to the body.
func (wr *V1Writer) writeU32(v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return wr.writeBodyBytes(b[:])
}

// writeU64 writes a little-endian u64 to the body.
func (wr *V1Writer) writeU64(v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	return wr.writeBodyBytes(b[:])
}

// writeF32 writes a little-endian f32 to the body.
func (wr *V1Writer) writeF32(v float32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
	return wr.writeBodyBytes(b[:])
}

// writeBF16s writes a slice of bfloat16 (u16 each) to the
// body. The values are stored as little-endian u16s, two
// bytes per element.
func (wr *V1Writer) writeBF16s(v []uint16) error {
	if len(v) == 0 {
		return nil
	}
	buf := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(buf[2*i:2*i+2], x)
	}
	return wr.writeBodyBytes(buf)
}

// writeF32s writes a slice of float32 to the body. Each
// element is 4 bytes, little-endian IEEE 754.
func (wr *V1Writer) writeF32s(v []float32) error {
	if len(v) == 0 {
		return nil
	}
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[4*i:4*i+4], math.Float32bits(x))
	}
	return wr.writeBodyBytes(buf)
}

// writeString writes a u32 length + u8 data to the body.
func (wr *V1Writer) writeString(s string) error {
	if err := wr.writeU32(uint32(len(s))); err != nil {
		return err
	}
	if len(s) == 0 {
		return nil
	}
	return wr.writeBodyBytes([]byte(s))
}

// writeBodyByte writes a single byte to both w and bodyHasher.
func (wr *V1Writer) writeBodyByte(b byte) error {
	if _, err := wr.bodyHasher.Write([]byte{b}); err != nil {
		return err
	}
	if _, err := wr.w.Write([]byte{b}); err != nil {
		return err
	}
	return nil
}

// writeBodyBytes writes b to both w and bodyHasher.
func (wr *V1Writer) writeBodyBytes(b []byte) error {
	if _, err := wr.bodyHasher.Write(b); err != nil {
		return err
	}
	if _, err := wr.w.Write(b); err != nil {
		return err
	}
	return nil
}

// ToOptimState converts a slice of AdamState (the v1
// format) into an optim.State (the existing public
// contract). Used by Transformer.Load to rehydrate the
// optimizer state.
//
// The wire values are float32 (the on-disk format and the
// in-memory master after the bf16 refactor). They are
// stored in state.Buffers as-is.
func ToOptimState(kind string, step uint64, states []AdamState) (*optim.State, error) {
	if kind != "adam" {
		return nil, fmt.Errorf("serialize: unsupported optimizer kind %q (v1 supports adam only)", kind)
	}
	buf := make(map[string][]float32, len(states)*2)
	for _, s := range states {
		mCopy := make([]float32, len(s.M))
		copy(mCopy, s.M)
		vCopy := make([]float32, len(s.V))
		copy(vCopy, s.V)
		buf["m."+s.Name] = mCopy
		buf["v."+s.Name] = vCopy
	}
	return &optim.State{
		Kind:    kind,
		Buffers: buf,
		Step:    int(step),
	}, nil
}
