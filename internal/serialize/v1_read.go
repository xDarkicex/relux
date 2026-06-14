package serialize

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"

	"github.com/xDarkicex/relux/internal/optim"
)

// V1Header is the parsed 32-byte v1 file header.
type V1Header struct {
	Magic       [4]byte
	Version     uint16
	Flags       uint16
	NumLayers   uint32
	TotalParams uint64
	HeaderCRC32 uint32
	Reserved    [8]byte
}

// V1Reader reads a .relux v1 file produced by V1Writer. It
// verifies the header's CRC32 on construction, then streams
// the body through a SHA-256 hasher so the footer can be
// verified on ReadFooter.
type V1Reader struct {
	r         io.Reader
	header    V1Header
	bodyHasher hash.Hash
	bytesRead int64
}

// NewV1Reader reads and verifies the 32-byte v1 header. The
// returned reader is positioned to read the first body byte
// from r. The caller is expected to then drive the reader
// (ReadArchEntry, ReadWeight, ReadOptimizerState) and finish
// with ReadFooter.
func NewV1Reader(r io.Reader) (*V1Reader, error) {
	var hdr V1Header
	if _, err := io.ReadFull(r, hdr.Magic[:]); err != nil {
		return nil, fmt.Errorf("v1: short header (magic): %w", err)
	}
	if !bytes.Equal(hdr.Magic[:], V1Magic[:]) {
		return nil, fmt.Errorf("v1: bad magic %q, want %q", hdr.Magic[:], V1Magic[:])
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.Version); err != nil {
		return nil, fmt.Errorf("v1: short header (version): %w", err)
	}
	if hdr.Version != V1Version {
		return nil, fmt.Errorf("v1: unsupported version %d (this build supports %d)", hdr.Version, V1Version)
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.Flags); err != nil {
		return nil, fmt.Errorf("v1: short header (flags): %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.NumLayers); err != nil {
		return nil, fmt.Errorf("v1: short header (num_layers): %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr.TotalParams); err != nil {
		return nil, fmt.Errorf("v1: short header (total_params): %w", err)
	}
	// At this point we've read 4+2+2+4+8 = 20 bytes. Read
	// the CRC32 and verify it against the 20 bytes.
	if err := binary.Read(r, binary.LittleEndian, &hdr.HeaderCRC32); err != nil {
		return nil, fmt.Errorf("v1: short header (crc32): %w", err)
	}
	// Rebuild the 20-byte pre-CRC region.
	var pre [20]byte
	copy(pre[0:4], hdr.Magic[:])
	binary.LittleEndian.PutUint16(pre[4:6], hdr.Version)
	binary.LittleEndian.PutUint16(pre[6:8], hdr.Flags)
	binary.LittleEndian.PutUint32(pre[8:12], hdr.NumLayers)
	binary.LittleEndian.PutUint64(pre[12:20], hdr.TotalParams)
	if crc32.ChecksumIEEE(pre[:]) != hdr.HeaderCRC32 {
		return nil, errors.New("v1: header CRC32 mismatch — file is corrupted or not a v1 .relux file")
	}
	// Skip the 8-byte reserved tail.
	if _, err := io.ReadFull(r, hdr.Reserved[:]); err != nil {
		return nil, fmt.Errorf("v1: short header (reserved): %w", err)
	}
	// Wrap the rest of r in a tee'd reader so any body
	// reads are also fed to the SHA-256 hasher.
	hasher := sha256.New()
	teeR := &teeReader{src: r, hasher: hasher}
	return &V1Reader{r: teeR, header: hdr, bodyHasher: hasher}, nil
}

// Header returns the parsed header. The caller can use
// NumLayers and TotalParams to preallocate, and read
// magic/version to verify.
func (r *V1Reader) Header() V1Header { return r.header }

// ReadArchEntry reads one arch entry. Returns the tag, the
// fixed-width dims, the trailing f32 payload, and the
// per-layer canonical name (used by the v1 writer for the
// arch block, but the reader doesn't depend on names; it
// reconstructs them from the layer type and arch position).
func (r *V1Reader) ReadArchEntry() (tag uint8, dims []uint32, floats []float32, err error) {
	tag, err = r.readU8()
	if err != nil {
		return 0, nil, nil, err
	}
	switch tag {
	case LayerTagEmbedding:
		// [u32 vocabSize, u32 dModel]
		vocab, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		d, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{vocab, d}, nil, nil
	case LayerTagRMSNorm:
		// [u32 dModel, f32 eps]
		d, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		eps, err := r.readF32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{d}, []float32{eps}, nil
	case LayerTagMHA:
		// [u32 dModel, u32 numHeads, u32 numKVHeads, u32 headDim]
		d, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		nh, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		nk, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		hd, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{d, nh, nk, hd}, nil, nil
	case LayerTagMLP:
		// [u32 dModel, u32 dFF]
		d, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		dff, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{d, dff}, nil, nil
	case LayerTagRoPE:
		// [u32 headDim, u32 maxSeqLen, f32 base]
		hd, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		msl, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		base, err := r.readF32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{hd, msl}, []float32{base}, nil
	case LayerTagLinear:
		// [u32 inDim, u32 outDim]
		in, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		out, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{in, out}, nil, nil
	case LayerTagMLA:
		// [u32 dModel, u32 numHeads, u32 dC, u32 dHR]
		d, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		nh, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		dc, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		dhr, err := r.readU32()
		if err != nil {
			return 0, nil, nil, err
		}
		return tag, []uint32{d, nh, dc, dhr}, nil, nil
	default:
		return 0, nil, nil, fmt.Errorf("v1: unknown arch tag %d at offset %d", tag, r.bytesRead)
	}
}

// ReadWeight reads one weight: a u32 element count and
// the bfloat16 data. The returned slice is in bfloat16
// (u16 per element); callers widen to float32 or float64
// as needed.
func (r *V1Reader) ReadWeight() ([]uint16, error) {
	length, err := r.readU32()
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return nil, nil
	}
	data := make([]uint16, length)
	if err := r.readBF16s(data); err != nil {
		return nil, err
	}
	return data, nil
}

// ReadWeightF32 reads one weight as float32. Used for the
// Adam optimizer state block (m and v are float32 in
// memory and on disk).
func (r *V1Reader) ReadWeightF32() ([]float32, error) {
	length, err := r.readU32()
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return nil, nil
	}
	data := make([]float32, length)
	if err := r.readF32s(data); err != nil {
		return nil, err
	}
	return data, nil
}

// ReadOptimizerState reads the optimizer state block. v1
// supports Adam only.
func (r *V1Reader) ReadOptimizerState() (kind string, step uint64, states []AdamState, err error) {
	kind, err = r.readString()
	if err != nil {
		return "", 0, nil, err
	}
	if step, err = r.readU64(); err != nil {
		return "", 0, nil, err
	}
	var n uint32
	if n, err = r.readU32(); err != nil {
		return "", 0, nil, err
	}
	states = make([]AdamState, 0, n)
	for i := uint32(0); i < n; i++ {
		name, err := r.readString()
		if err != nil {
			return "", 0, nil, err
		}
		m, err := r.ReadWeightF32()
		if err != nil {
			return "", 0, nil, err
		}
		v, err := r.ReadWeightF32()
		if err != nil {
			return "", 0, nil, err
		}
		states = append(states, AdamState{Name: name, M: m, V: v})
	}
	return kind, step, states, nil
}

// ReadFooter reads the 32-byte SHA-256 footer and verifies
// it matches the body hash. Must be the last call.
func (r *V1Reader) ReadFooter() error {
	sum := r.bodyHasher.Sum(nil)
	var footer [V1FooterSize]byte
	if _, err := io.ReadFull(r.r, footer[:]); err != nil {
		return fmt.Errorf("v1: short footer: %w", err)
	}
	if !bytes.Equal(footer[:], sum) {
		return fmt.Errorf("v1: body SHA-256 mismatch — body is corrupted or tampered (expected %x, got %x)", sum, footer[:])
	}
	return nil
}

// readU8 reads a single byte.
func (r *V1Reader) readU8() (uint8, error) {
	var b [1]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	r.bytesRead += 1
	return b[0], nil
}

// readU16 reads a little-endian u16.
func (r *V1Reader) readU16() (uint16, error) {
	var b [2]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	r.bytesRead += 2
	return binary.LittleEndian.Uint16(b[:]), nil
}

// readU32 reads a little-endian u32.
func (r *V1Reader) readU32() (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	r.bytesRead += 4
	return binary.LittleEndian.Uint32(b[:]), nil
}

// readU64 reads a little-endian u64.
func (r *V1Reader) readU64() (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r.r, b[:]); err != nil {
		return 0, err
	}
	r.bytesRead += 8
	return binary.LittleEndian.Uint64(b[:]), nil
}

// readF32 reads a little-endian f32.
func (r *V1Reader) readF32() (float32, error) {
	v, err := r.readU32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(v), nil
}

// readBF16s reads a slice of bfloat16. Each element is a
// little-endian u16 in the on-disk format; the
// destination slice is the same u16 representation (no
// conversion — the wire format IS the bfloat16 bits).
func (r *V1Reader) readBF16s(dst []uint16) error {
	buf := make([]byte, 2*len(dst))
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return err
	}
	r.bytesRead += int64(len(buf))
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint16(buf[2*i : 2*i+2])
	}
	return nil
}

// readF32s reads a slice of float32. Each element is 4
// bytes, little-endian IEEE 754.
func (r *V1Reader) readF32s(dst []float32) error {
	buf := make([]byte, 4*len(dst))
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return err
	}
	r.bytesRead += int64(len(buf))
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i : 4*i+4]))
	}
	return nil
}

// readString reads a u32 length + u8 data string.
func (r *V1Reader) readString() (string, error) {
	n, err := r.readU32()
	if err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return "", err
	}
	r.bytesRead += int64(n)
	return string(buf), nil
}

// teeReader wraps an io.Reader so that all reads are also
// fed to a hash. It satisfies io.Reader.
type teeReader struct {
	src    io.Reader
	hasher hash.Hash
}

func (t *teeReader) Read(p []byte) (int, error) {
	n, err := t.src.Read(p)
	if n > 0 {
		if _, herr := t.hasher.Write(p[:n]); herr != nil {
			return n, herr
		}
	}
	return n, err
}

// FromOptimState extracts Adam state from an optim.State.
// The result is the v1 wire format: float32 slices for m
// and v per param. The kind is the optim.State.Kind
// (typically "adam"). step is the Adam timestep.
//
// Param names are the ones the writer emits: raw short
// names (e.g. "mha.Wq", "rmsnorm.gamma") — the layer
// prefix from the live params is preserved verbatim
// because the existing optim.State already uses the
// same naming.
func FromOptimState(state *optim.State) (kind string, step uint64, states []AdamState) {
	if state == nil {
		return "", 0, nil
	}
	step = uint64(state.Step)
	// Group buffers by param name. Adam uses m.<name> and
	// v.<name> prefixes.
	byName := make(map[string]*AdamState)
	keys := make([]string, 0, len(state.Buffers))
	for k, buf := range state.Buffers {
		var name, which string
		switch {
		case len(k) > 2 && k[:2] == "m.":
			name, which = k[2:], "m"
		case len(k) > 2 && k[:2] == "v.":
			name, which = k[2:], "v"
		default:
			continue
		}
		entry, ok := byName[name]
		if !ok {
			entry = &AdamState{Name: name}
			byName[name] = entry
			keys = append(keys, name)
		}
		// The in-memory and on-disk formats are both
		// float32 — copy through.
		f32 := make([]float32, len(buf))
		copy(f32, buf)
		if which == "m" {
			entry.M = f32
		} else {
			entry.V = f32
		}
	}
	for _, name := range keys {
		states = append(states, *byName[name])
	}
	return state.Kind, step, states
}
