// Package loader reads external model formats (SafeTensors, GGUF) and
// maps weights into relux's optim.Param slots.
//
// SafeTensors format:
//   [u64 LE] header_size (bytes)
//   [header_size bytes] JSON header
//   [rest of file] raw tensor data
//
// JSON header is an object: each key is a tensor name, value is
// { "dtype": "F32"|"F16"|"BF16"|..., "shape": [int, ...],
//   "data_offsets": [start, end] }.
// The reserved key "__metadata__" is a string map (ignored for now).
package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"

	"github.com/xDarkicex/relux/internal/alloc"
)

// SafeTensorsDType is the on-disk dtype tag in the JSON header.
type SafeTensorsDType string

const (
	SafeTensorsF16  SafeTensorsDType = "F16"
	SafeTensorsF32  SafeTensorsDType = "F32"
	SafeTensorsF64  SafeTensorsDType = "F64"
	SafeTensorsBF16 SafeTensorsDType = "BF16"
	SafeTensorsI32  SafeTensorsDType = "I32"
	SafeTensorsI64  SafeTensorsDType = "I64"
	SafeTensorsU8   SafeTensorsDType = "U8"
	SafeTensorsBool SafeTensorsDType = "BOOL"
)

// SafeTensorInfo holds the metadata for one tensor in the file.
type SafeTensorInfo struct {
	DType       SafeTensorsDType `json:"dtype"`
	Shape       []int            `json:"shape"`
	DataOffsets [2]int64         `json:"data_offsets"`
}

// SafeTensorsHeader is the parsed JSON header. The map is keyed by
// tensor name. __metadata__ is the reserved metadata key.
type SafeTensorsHeader struct {
	Metadata map[string]string            `json:"__metadata__"`
	Tensors  map[string]SafeTensorInfo    `json:"-"`
	Raw      map[string]json.RawMessage   `json:"-"`
}

// NumBytes returns the byte count this tensor occupies on disk,
// computed from data_offsets.
func (s SafeTensorInfo) NumBytes() int64 {
	return s.DataOffsets[1] - s.DataOffsets[0]
}

// BytesPerElement returns the wire size of one element of dtype.
func (s SafeTensorInfo) BytesPerElement() int {
	switch s.DType {
	case SafeTensorsF64, SafeTensorsI64:
		return 8
	case SafeTensorsF32, SafeTensorsI32:
		return 4
	case SafeTensorsF16, SafeTensorsBF16:
		return 2
	case SafeTensorsU8, SafeTensorsBool:
		return 1
	default:
		return 0
	}
}

// NumElements returns the product of shape dimensions.
func (s SafeTensorInfo) NumElements() int {
	n := 1
	for _, d := range s.Shape {
		n *= d
	}
	return n
}

// SafeTensorsReader parses and reads a SafeTensors file. The file
// handle stays open and the data region is read on demand via Seek/Read.
type SafeTensorsReader struct {
	file       *os.File
	header     SafeTensorsHeader
	dataOffset int64 // file position of the first byte of tensor data
}

// NewSafeTensorsReader opens a .safetensors file, parses the header,
// and positions the file pointer for tensor reads.
func NewSafeTensorsReader(path string) (*SafeTensorsReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open: %w", err)
	}

	// 1. Read the 8-byte little-endian header size.
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: read header size: %w", err)
	}

	// 2. Read the JSON header.
	headerBytes := alloc.ByteSlice(int(headerSize))
	if _, err := readFull(f, headerBytes); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: read header bytes: %w", err)
	}

	// 3. Parse JSON. First pass: generic map. Then unmarshal each
	//    non-metadata entry into SafeTensorInfo.
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	hdr := SafeTensorsHeader{
		Metadata: map[string]string{},
		Tensors:  map[string]SafeTensorInfo{},
		Raw:      raw,
	}

	for k, v := range raw {
		if k == "__metadata__" {
			_ = json.Unmarshal(v, &hdr.Metadata)
			continue
		}
		var info SafeTensorInfo
		if err := json.Unmarshal(v, &info); err != nil {
			f.Close()
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", k, err)
		}
		hdr.Tensors[k] = info
	}

	// 4. Compute the data region offset (current position after the header).
	dataOff, err := f.Seek(0, 1) // SeekCurrent
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: get data offset: %w", err)
	}

	return &SafeTensorsReader{
		file:       f,
		header:     hdr,
		dataOffset: dataOff,
	}, nil
}

// Close releases the file handle.
func (r *SafeTensorsReader) Close() error {
	if r.file != nil {
		err := r.file.Close()
		r.file = nil
		return err
	}
	return nil
}

// TensorNames returns the names of all tensors in the file.
func (r *SafeTensorsReader) TensorNames() []string {
	names := make([]string, 0, len(r.header.Tensors))
	for k := range r.header.Tensors {
		names = append(names, k)
	}
	return names
}

// TensorInfo returns the metadata for a single tensor.
func (r *SafeTensorsReader) TensorInfo(name string) (SafeTensorInfo, error) {
	info, ok := r.header.Tensors[name]
	if !ok {
		return SafeTensorInfo{}, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	return info, nil
}

// ReadTensorData reads the raw bytes of one tensor. The returned
// slice is a freshly allocated []byte (Go heap); the OS page cache
// serves the actual read.
func (r *SafeTensorsReader) ReadTensorData(name string) ([]byte, error) {
	info, err := r.TensorInfo(name)
	if err != nil {
		return nil, err
	}
	start := r.dataOffset + info.DataOffsets[0]
	end := r.dataOffset + info.DataOffsets[1]
	size := end - start
	if size < 0 {
		return nil, fmt.Errorf("safetensors: tensor %q has negative data_offsets range", name)
	}
	if _, err := r.file.Seek(start, 0); err != nil {
		return nil, fmt.Errorf("safetensors: seek %q: %w", name, err)
	}
	buf := alloc.ByteSlice(int(size))
	if _, err := readFull(r.file, buf); err != nil {
		return nil, fmt.Errorf("safetensors: read %q: %w", name, err)
	}
	return buf, nil
}

// ReadTensorDataInto reads tensor bytes into a caller-provided buffer.
// Returns the number of bytes actually read.
func (r *SafeTensorsReader) ReadTensorDataInto(name string, dst []byte) (int, error) {
	info, err := r.TensorInfo(name)
	if err != nil {
		return 0, err
	}
	start := r.dataOffset + info.DataOffsets[0]
	end := r.dataOffset + info.DataOffsets[1]
	size := end - start
	if size > int64(len(dst)) {
		return 0, fmt.Errorf("safetensors: tensor %q needs %d bytes, dst has %d", name, size, len(dst))
	}
	if _, err := r.file.Seek(start, 0); err != nil {
		return 0, err
	}
	if _, err := readFull(r.file, dst[:size]); err != nil {
		return 0, err
	}
	return int(size), nil
}

// readFull reads exactly len(buf) bytes or returns an error.
func readFull(f *os.File, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := f.Read(buf[total:])
		if n > 0 {
			total += n
		}
		if err != nil {
			if total == len(buf) {
				return total, nil
			}
			return total, err
		}
	}
	return total, nil
}
