package loader

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/xDarkicex/memory"
	"github.com/xDarkicex/relux/internal/alloc"
)

// GGUF format constants. GGUF v3 is the llama.cpp standard.
const (
	ggufMagicLE   uint32 = 0x46554747 // "GGUF" little-endian
	ggufMagicBE   uint32 = 0x47475546 // "GGUF" big-endian
	ggufVersionV1 uint32 = 1
	ggufVersionV2 uint32 = 2
	ggufVersionV3 uint32 = 3
)

// Default alignment for the tensor data section. Overrideable via
// "general.alignment" metadata key.
const ggufDefaultAlignment = 32

// GGUFValueType identifies a metadata value's wire type.
type GGUFValueType uint32

const (
	GGUFValueUint8   GGUFValueType = 0
	GGUFValueInt8    GGUFValueType = 1
	GGUFValueUint16  GGUFValueType = 16
	GGUFValueInt16   GGUFValueType = 17
	GGUFValueUint32  GGUFValueType = 2
	GGUFValueInt32   GGUFValueType = 3
	GGUFValueFloat32 GGUFValueType = 4
	GGUFValueBool    GGUFValueType = 5
	GGUFValueString  GGUFValueType = 6
	GGUFValueArray   GGUFValueType = 7
	GGUFValueUint64  GGUFValueType = 8
	GGUFValueInt64   GGUFValueType = 9
	GGUFValueFloat64 GGUFValueType = 10
)

// GGUFTensorInfo describes one tensor's location in the file.
type GGUFTensorInfo struct {
	Name       string
	Dims       []uint64 // reversed relative to PyTorch convention
	DType      GGMLType
	Offset     uint64 // relative to start of tensor data section
	NumBytes   uint64 // size in bytes
	NumElement uint64
}

// GGUFReader parses and reads a GGUF v3 file. The tensor data region
// is mmap'd for zero-copy reads.
type GGUFReader struct {
	file       *os.File
	mmapData   []byte // memory.MmapFile on the tensor data region
	header     ggufHeader
	metadata   map[string]interface{}
	tensors    map[string]GGUFTensorInfo
	dataOffset int64 // file position of first tensor data byte
}

// ggufHeader is the parsed GGUF file header.
type ggufHeader struct {
	Magic     uint32
	Version   uint32
	Tensors   uint64
	MetaCount uint64
}

// NewGGUFReader opens a .gguf file, parses header + metadata + tensor
// infos, and mmap's the tensor data region.
func NewGGUFReader(path string) (*GGUFReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open: %w", err)
	}

	// 1. Read the 24-byte header.
	var hdr ggufHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr.Magic); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	var order binary.ByteOrder = binary.LittleEndian
	switch hdr.Magic {
	case ggufMagicLE:
		order = binary.LittleEndian
	case ggufMagicBE:
		order = binary.BigEndian
		// Re-read the rest in big-endian.
	default:
		f.Close()
		return nil, fmt.Errorf("gguf: bad magic 0x%08X", hdr.Magic)
	}

	if err := binary.Read(f, order, &hdr.Version); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if hdr.Version < ggufVersionV1 || hdr.Version > ggufVersionV3 {
		f.Close()
		return nil, fmt.Errorf("gguf: unsupported version %d (1-3 supported)", hdr.Version)
	}
	if err := binary.Read(f, order, &hdr.Tensors); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	if err := binary.Read(f, order, &hdr.MetaCount); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read meta count: %w", err)
	}

	// 2. Read metadata KVs.
	meta := make(map[string]interface{}, hdr.MetaCount)
	alignment := ggufDefaultAlignment
	for i := uint64(0); i < hdr.MetaCount; i++ {
		k, v, err := readMetadataKV(f, order)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: read metadata %d: %w", i, err)
		}
		meta[k] = v
		if k == "general.alignment" {
			if a, ok := v.(uint32); ok {
				alignment = int(a)
			}
		}
	}

	// 3. Read tensor infos.
	tensors := make(map[string]GGUFTensorInfo, hdr.Tensors)
	for i := uint64(0); i < hdr.Tensors; i++ {
		t, err := readTensorInfo(f, order)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: read tensor info %d: %w", i, err)
		}
		tensors[t.Name] = t
	}

	// 4. Compute data region offset (file position aligned up).
	pos, err := f.Seek(0, 1)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: get pos: %w", err)
	}
	alignedPos := alignUp(pos, int64(alignment))

	// 5. Compute total file size for the mmap.
	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: stat: %w", err)
	}
	mmapLen := stat.Size() - alignedPos
	if mmapLen < 0 {
		f.Close()
		return nil, fmt.Errorf("gguf: file too small for tensor data")
	}

	// 6. Mmap the data region.
	mmapData, err := memory.MmapFileReadOnly(int(f.Fd()), alignedPos, int(mmapLen))
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: mmap: %w", err)
	}

	return &GGUFReader{
		file:       f,
		mmapData:   mmapData,
		header:     hdr,
		metadata:   meta,
		tensors:    tensors,
		dataOffset: alignedPos,
	}, nil
}

// Close releases the mmap and file handle.
func (r *GGUFReader) Close() error {
	if r.mmapData != nil {
		_ = memory.Munmap(r.mmapData)
		r.mmapData = nil
	}
	if r.file != nil {
		err := r.file.Close()
		r.file = nil
		return err
	}
	return nil
}

// Metadata returns the parsed metadata map.
func (r *GGUFReader) Metadata() map[string]interface{} {
	return r.metadata
}

// TensorNames returns all tensor names. The result is a Go-heap
// []string (Go has no off-heap string slice support). For a
// 7B-param model this is ~250 names ≈ 8 KB; the cost is paid
// once at file open.
func (r *GGUFReader) TensorNames() []string {
	names := make([]string, 0, len(r.tensors))
	for k := range r.tensors {
		names = append(names, k)
	}
	return names
}

// TensorInfo returns the metadata for one tensor.
func (r *GGUFReader) TensorInfo(name string) (GGUFTensorInfo, error) {
	t, ok := r.tensors[name]
	if !ok {
		return GGUFTensorInfo{}, fmt.Errorf("gguf: tensor %q not found", name)
	}
	return t, nil
}

// ReadTensorData returns a zero-copy slice into the mmap'd region.
// The slice is valid until Close() is called.
func (r *GGUFReader) ReadTensorData(name string) ([]byte, error) {
	t, err := r.TensorInfo(name)
	if err != nil {
		return nil, err
	}
	start := int(t.Offset)
	end := start + int(t.NumBytes)
	if start < 0 || end > len(r.mmapData) {
		return nil, fmt.Errorf("gguf: tensor %q out of mmap bounds", name)
	}
	return r.mmapData[start:end], nil
}

// ReadTensorDataDequantized reads raw bytes, dequantizes to f32, and
// returns the freshly allocated f32 slice. Uses alloc.Float32 to keep
// the result off-heap.
//
// If dtype is non-quantized (F32/F16/BF16), the raw bytes are either
// cast directly or widened to f32 with no copy beyond the output.
func (r *GGUFReader) ReadTensorDataDequantized(name string) ([]float32, error) {
	t, err := r.TensorInfo(name)
	if err != nil {
		return nil, err
	}
	raw, err := r.ReadTensorData(name)
	if err != nil {
		return nil, err
	}
	out, err := allocFloat32(int(t.NumElement))
	if err != nil {
		return nil, err
	}
	if _, err := Dequantize(raw, t.DType, int(t.NumElement), out); err != nil {
		return nil, err
	}
	return out, nil
}

// Header returns the parsed file header.
func (r *GGUFReader) Header() ggufHeader { return r.header }

// --- reading primitives ---

// readString reads a uint64-length-prefixed little-endian or big-endian
// string.
func readString(f *os.File, order binary.ByteOrder) (string, error) {
	var l uint64
	if err := binary.Read(f, order, &l); err != nil {
		return "", err
	}
	// Sanity: cap at 1 MB for metadata strings.
	if l > 1<<20 {
		return "", fmt.Errorf("string length %d exceeds 1MB", l)
	}
	buf := alloc.ByteSlice(int(l))
	if _, err := readFull(f, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readMetadataKV reads one metadata key-value pair.
func readMetadataKV(f *os.File, order binary.ByteOrder) (string, interface{}, error) {
	k, err := readString(f, order)
	if err != nil {
		return "", nil, fmt.Errorf("key: %w", err)
	}
	var vt uint32
	if err := binary.Read(f, order, &vt); err != nil {
		return "", nil, fmt.Errorf("value type: %w", err)
	}
	v, err := readMetadataValue(f, order, GGUFValueType(vt))
	if err != nil {
		return k, nil, fmt.Errorf("value: %w", err)
	}
	return k, v, nil
}

func readMetadataValue(f *os.File, order binary.ByteOrder, vt GGUFValueType) (interface{}, error) {
	switch vt {
	case GGUFValueUint8:
		var v uint8
		return v, binary.Read(f, order, &v)
	case GGUFValueInt8:
		var v int8
		return v, binary.Read(f, order, &v)
	case GGUFValueUint16:
		var v uint16
		return v, binary.Read(f, order, &v)
	case GGUFValueInt16:
		var v int16
		return v, binary.Read(f, order, &v)
	case GGUFValueUint32:
		var v uint32
		return v, binary.Read(f, order, &v)
	case GGUFValueInt32:
		var v int32
		return v, binary.Read(f, order, &v)
	case GGUFValueFloat32:
		var v float32
		return v, binary.Read(f, order, &v)
	case GGUFValueBool:
		var v uint8
		if err := binary.Read(f, order, &v); err != nil {
			return false, err
		}
		return v != 0, nil
	case GGUFValueString:
		return readString(f, order)
	case GGUFValueArray:
		return readArrayValue(f, order)
	case GGUFValueUint64:
		var v uint64
		return v, binary.Read(f, order, &v)
	case GGUFValueInt64:
		var v int64
		return v, binary.Read(f, order, &v)
	case GGUFValueFloat64:
		var v float64
		return v, binary.Read(f, order, &v)
	default:
		return nil, fmt.Errorf("unknown value type %d", vt)
	}
}

func readArrayValue(f *os.File, order binary.ByteOrder) (interface{}, error) {
	var elemType uint32
	if err := binary.Read(f, order, &elemType); err != nil {
		return nil, err
	}
	var length uint64
	if err := binary.Read(f, order, &length); err != nil {
		return nil, err
	}
	if length > 100_000_000 {
		return nil, fmt.Errorf("array length %d too large", length)
	}
	vt := GGUFValueType(elemType)

	switch vt {
	case GGUFValueUint8:
		out := alloc.Uint8(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueInt8:
		out := alloc.Int8(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueUint16:
		out := alloc.Uint16(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueInt16:
		out := alloc.Int16(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueUint32:
		out := alloc.Uint32(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueInt32:
		out := alloc.Int32(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueFloat32:
		out := alloc.Float32(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueBool:
		out := alloc.Bool(int(length))
		buf := alloc.Uint8(int(length))
		if err := binary.Read(f, order, buf); err != nil {
			return nil, err
		}
		for i := range buf {
			out[i] = buf[i] != 0
		}
		alloc.Free(buf)
		return out, nil
	case GGUFValueString:
		// Go's runtime does not support off-heap []string
		// (string headers are 16 bytes with a separate data
		// pointer; backing them with a []byte requires a
		// string allocation per element). The dominant caller
		// here is the GGUF metadata key "tokenizer.ggml.tokens"
		// which is a vocab list of 32k–200k entries (~3 MB
		// Go-heap peak). We accept this one-time cost because:
		//  (a) it happens at file-open, not in the hot path;
		//  (b) relux uses the vocab list only to count entries
		//      and then discards it.
		// Future work: add a `readStringArraySkip` variant that
		// fast-forwards through the array without materializing.
		out := make([]string, length)
		for i := uint64(0); i < length; i++ {
			s, err := readString(f, order)
			if err != nil {
				return nil, err
			}
			out[i] = s
		}
		return out, nil
	case GGUFValueInt64:
		out := alloc.Int64(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	case GGUFValueFloat64:
		out := alloc.Float64(int(length))
		if err := binary.Read(f, order, out); err != nil {
			return nil, err
		}
		return out, nil
	default:
		return nil, fmt.Errorf("array element type %d unhandled", vt)
	}
}

// readTensorInfo reads one tensor's metadata block.
func readTensorInfo(f *os.File, order binary.ByteOrder) (GGUFTensorInfo, error) {
	name, err := readString(f, order)
	if err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("name: %w", err)
	}
	var ndims uint32
	if err := binary.Read(f, order, &ndims); err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("ndims: %w", err)
	}
	if ndims > 8 {
		return GGUFTensorInfo{}, fmt.Errorf("too many dims: %d", ndims)
	}
	dims := alloc.Uint64(int(ndims))
	for i := uint32(0); i < ndims; i++ {
		if err := binary.Read(f, order, &dims[i]); err != nil {
			return GGUFTensorInfo{}, fmt.Errorf("dim %d: %w", i, err)
		}
	}
	var dtype uint32
	if err := binary.Read(f, order, &dtype); err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("dtype: %w", err)
	}
	var offset uint64
	if err := binary.Read(f, order, &offset); err != nil {
		return GGUFTensorInfo{}, fmt.Errorf("offset: %w", err)
	}

	var nElem uint64 = 1
	for _, d := range dims {
		nElem *= d
	}
	return GGUFTensorInfo{
		Name:       name,
		Dims:       dims,
		DType:      GGMLType(dtype),
		Offset:     offset,
		NumElement: nElem,
		NumBytes:   uint64(ggmlRequiredBytes(GGMLType(dtype), int(nElem))),
	}, nil
}

// alignUp rounds v up to the next multiple of align.
func alignUp(v, align int64) int64 {
	if align <= 0 {
		return v
	}
	return (v + align - 1) & ^(align - 1)
}

// allocFloat32 is defined in mapper.go as a forward declaration
// and wired in mapper.go's init() to alloc.Float32.
