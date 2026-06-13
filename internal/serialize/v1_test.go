package serialize

import (
	"bytes"
	"encoding/gob"
	"errors"
	"io"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

// TestV1WriterReaderRoundTrip exercises the low-level
// V1Writer and V1Reader: write an arch + weight block,
// read it back, verify the dims and data match exactly.
//
// Weights and Adam state are stored as bfloat16 in the v1
// format; the round-trip is bit-exact for the wire format,
// but the on-the-wire representation is the bf16 encoding
// of the input float32 (or float64 for Adam state).
func TestV1WriterReaderRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	// 3 entries: 1 RoPE, 1 RMSNorm, 1 Linear.
	const numEntries = 3
	// 4*2 = 8 weight elements (the Linear's W is 4*2).
	const totalParams = int64(8)
	wr := NewV1Writer(&buf, numEntries, totalParams)
	if err := wr.WriteHeader(); err != nil {
		t.Fatalf("WriteHeader: %v", err)
	}
	// Arch entry 1: RoPE (headDim=16, maxSeqLen=128, base=10000)
	if err := wr.WriteArchEntry(LayerTagRoPE,
		[]uint32{16, 128},
		[]float32{10000.0}); err != nil {
		t.Fatalf("WriteArchEntry RoPE: %v", err)
	}
	// Arch entry 2: RMSNorm (dModel=8, eps=1e-5)
	if err := wr.WriteArchEntry(LayerTagRMSNorm,
		[]uint32{8},
		[]float32{1e-5}); err != nil {
		t.Fatalf("WriteArchEntry RMSNorm: %v", err)
	}
	// Arch entry 3: Linear (in=4, out=2)
	if err := wr.WriteArchEntry(LayerTagLinear,
		[]uint32{4, 2}, nil); err != nil {
		t.Fatalf("WriteArchEntry Linear: %v", err)
	}
	// Weight: 8 bfloat16 values. Choose values that
	// round-trip exactly (use small ints).
	dataF32 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	data := transformer.BF16SliceFromF32(dataF32)
	if err := wr.WriteWeight(data); err != nil {
		t.Fatalf("WriteWeight: %v", err)
	}
	// Adam state: 1 param with 8 m, 8 v values. Adam state
	// is float32 on disk (not bf16) to avoid the
	// quantization-shock problem.
	mF32 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	vF32 := []float32{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}
	if err := wr.WriteOptimizerState("adam", 5, []AdamState{
		{Name: "W", M: mF32, V: vF32},
	}); err != nil {
		t.Fatalf("WriteOptimizerState: %v", err)
	}
	if _, err := wr.WriteFooter(); err != nil {
		t.Fatalf("WriteFooter: %v", err)
	}
	// Read it back.
	rdr, err := NewV1Reader(&buf)
	if err != nil {
		t.Fatalf("NewV1Reader: %v", err)
	}
	if rdr.Header().NumLayers != numEntries {
		t.Fatalf("header numLayers=%d, want %d", rdr.Header().NumLayers, numEntries)
	}
	// Arch 1: RoPE
	tag, dims, floats, err := rdr.ReadArchEntry()
	if err != nil {
		t.Fatalf("ReadArchEntry RoPE: %v", err)
	}
	if tag != LayerTagRoPE {
		t.Errorf("RoPE tag=%d, want %d", tag, LayerTagRoPE)
	}
	if dims[0] != 16 || dims[1] != 128 {
		t.Errorf("RoPE dims=%v, want {16, 128}", dims)
	}
	if floats[0] != 10000.0 {
		t.Errorf("RoPE base=%v, want 10000", floats[0])
	}
	// Arch 2: RMSNorm
	tag, dims, floats, err = rdr.ReadArchEntry()
	if err != nil {
		t.Fatalf("ReadArchEntry RMSNorm: %v", err)
	}
	if tag != LayerTagRMSNorm {
		t.Errorf("RMSNorm tag=%d, want %d", tag, LayerTagRMSNorm)
	}
	if dims[0] != 8 {
		t.Errorf("RMSNorm dModel=%d, want 8", dims[0])
	}
	if floats[0] != 1e-5 {
		t.Errorf("RMSNorm eps=%v, want 1e-5", floats[0])
	}
	// Arch 3: Linear
	tag, dims, _, err = rdr.ReadArchEntry()
	if err != nil {
		t.Fatalf("ReadArchEntry Linear: %v", err)
	}
	if tag != LayerTagLinear {
		t.Errorf("Linear tag=%d, want %d", tag, LayerTagLinear)
	}
	if dims[0] != 4 || dims[1] != 2 {
		t.Errorf("Linear dims=%v, want {4, 2}", dims)
	}
	// Weights — bit-exact round trip of the bf16 wire format.
	got, err := rdr.ReadWeight()
	if err != nil {
		t.Fatalf("ReadWeight: %v", err)
	}
	if len(got) != 8 {
		t.Fatalf("ReadWeight len=%d, want 8", len(got))
	}
	for i := range data {
		if got[i] != data[i] {
			t.Errorf("weight[%d]=%v, want %v", i, got[i], data[i])
		}
	}
	// Optimizer state.
	kind, step, states, err := rdr.ReadOptimizerState()
	if err != nil {
		t.Fatalf("ReadOptimizerState: %v", err)
	}
	if kind != "adam" {
		t.Errorf("kind=%q, want adam", kind)
	}
	if step != 5 {
		t.Errorf("step=%d, want 5", step)
	}
	if len(states) != 1 {
		t.Fatalf("states len=%d, want 1", len(states))
	}
	if states[0].Name != "W" {
		t.Errorf("states[0].Name=%q, want W", states[0].Name)
	}
	if len(states[0].M) != 8 {
		t.Errorf("states[0].M len=%d, want 8", len(states[0].M))
	}
	// Adam state is float32 on disk and in memory; the
	// round-trip is bit-exact.
	if states[0].M[0] != 1.0 {
		t.Errorf("states[0].M[0]=%v, want 1.0", states[0].M[0])
	}
	if states[0].V[0] != 0.5 {
		t.Errorf("states[0].V[0]=%v, want 0.5", states[0].V[0])
	}
	// Footer
	if err := rdr.ReadFooter(); err != nil {
		t.Fatalf("ReadFooter: %v", err)
	}
}

// TestV1HeaderCRC32Corruption flips a byte in the header
// (past magic and version) and verifies the reader
// rejects the file with a CRC32 mismatch error.
func TestV1HeaderCRC32Corruption(t *testing.T) {
	var buf bytes.Buffer
	wr := NewV1Writer(&buf, 0, 0)
	if err := wr.WriteHeader(); err != nil {
		t.Fatalf("WriteHeader: %v", err)
	}
	// We must at least emit an empty optimizer state and
	// the footer for a "complete" file.
	if err := wr.WriteOptimizerState("adam", 0, nil); err != nil {
		t.Fatalf("WriteOptimizerState: %v", err)
	}
	if _, err := wr.WriteFooter(); err != nil {
		t.Fatalf("WriteFooter: %v", err)
	}
	data := buf.Bytes()
	// Don't touch the magic (that would trigger the magic
	// check) or the version (that would trigger the
	// version check). Flip a byte in the flags field at
	// offset 6, which IS covered by the CRC32.
	data[6] ^= 0xFF
	_, err := NewV1Reader(bytes.NewReader(data))
	if err == nil {
		t.Fatalf("NewV1Reader: expected error, got nil")
	}
	if !contains(err.Error(), "CRC32") {
		t.Errorf("error %q does not mention CRC32", err)
	}
}

// TestV1BodySHA256Corruption flips a byte in the body
// (after the header) and verifies the reader rejects the
// file with a SHA-256 mismatch error.
func TestV1BodySHA256Corruption(t *testing.T) {
	var buf bytes.Buffer
	wr := NewV1Writer(&buf, 1, 4)
	if err := wr.WriteHeader(); err != nil {
		t.Fatalf("WriteHeader: %v", err)
	}
	if err := wr.WriteArchEntry(LayerTagRMSNorm, []uint32{4}, []float32{1e-5}); err != nil {
		t.Fatalf("WriteArchEntry: %v", err)
	}
	if err := wr.WriteWeight(transformer.BF16SliceFromF32([]float32{1, 2, 3, 4})); err != nil {
		t.Fatalf("WriteWeight: %v", err)
	}
	if err := wr.WriteOptimizerState("adam", 0, nil); err != nil {
		t.Fatalf("WriteOptimizerState: %v", err)
	}
	if _, err := wr.WriteFooter(); err != nil {
		t.Fatalf("WriteFooter: %v", err)
	}
	// Header is 32 bytes. Body starts at offset 32.
	// Inside the body: tag (1) + u32 (4) + f32 (4) = 9
	// bytes of arch. Then weight: u32 length (4) + 4 bf16
	// elements (8 bytes) = 12 bytes. So weight data starts
	// at body offset 9 + 4 = 13, file offset 45. Flip the
	// second bf16 element (2 bytes) at file offset 45+2=47.
	data := buf.Bytes()
	data[47] ^= 0xFF
	// Read header — should pass CRC32.
	rdr, err := NewV1Reader(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("NewV1Reader (header should pass): %v", err)
	}
	// Read arch entry.
	if _, _, _, err := rdr.ReadArchEntry(); err != nil {
		t.Fatalf("ReadArchEntry: %v", err)
	}
	// Read weight.
	if _, err := rdr.ReadWeight(); err != nil {
		t.Fatalf("ReadWeight: %v", err)
	}
	// Optimizer state.
	if _, _, _, err := rdr.ReadOptimizerState(); err != nil {
		t.Fatalf("ReadOptimizerState: %v", err)
	}
	// Footer should fail with SHA-256 mismatch.
	if err := rdr.ReadFooter(); err == nil {
		t.Errorf("ReadFooter: expected SHA-256 mismatch, got nil")
	} else if !contains(err.Error(), "SHA-256") {
		t.Errorf("ReadFooter error %q does not mention SHA-256", err)
	}
}

// TestV1VersionMismatch writes a header with an
// unsupported version and verifies the reader rejects it.
func TestV1VersionMismatch(t *testing.T) {
	var buf bytes.Buffer
	wr := NewV1Writer(&buf, 0, 0)
	if err := wr.WriteHeader(); err != nil {
		t.Fatalf("WriteHeader: %v", err)
	}
	if err := wr.WriteOptimizerState("adam", 0, nil); err != nil {
		t.Fatalf("WriteOptimizerState: %v", err)
	}
	if _, err := wr.WriteFooter(); err != nil {
		t.Fatalf("WriteFooter: %v", err)
	}
	// Bump the version byte to 99.
	data := buf.Bytes()
	data[5] = 99
	_, err := NewV1Reader(bytes.NewReader(data))
	if err == nil {
		t.Fatalf("NewV1Reader: expected error, got nil")
	}
	if !contains(err.Error(), "version") {
		t.Errorf("error %q does not mention version", err)
	}
}

// TestV1MagicMismatch checks the reader rejects a file
// whose first 4 bytes are not "RELV".
func TestV1MagicMismatch(t *testing.T) {
	bad := []byte{'X', 'X', 'X', 'X', 0, 0, 0, 0, 0, 0, 0, 0}
	_, err := NewV1Reader(bytes.NewReader(bad))
	if err == nil {
		t.Fatalf("expected magic error, got nil")
	}
	if !contains(err.Error(), "magic") {
		t.Errorf("error %q does not mention magic", err)
	}
}

// TestV0GobBackCompat verifies the v0 reader (gob
// wrapper) reads a NetworkSnapshot correctly. This is the
// smoke test for the v0 back compat path.
func TestV0GobBackCompat(t *testing.T) {
	// Build a snapshot, gob-encode it, decode via V0Read.
	snap := &NetworkSnapshot{
		InputSize: 3,
		Layers: []LayerData{
			{
				Weights:        []uint16{0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0}, // 1, 2, 3, 4, 5, 6
				WeightShape:    [2]int{2, 3},
				Biases:         []uint16{0, 0, 0},
				ActivationName: "relu",
			},
		},
		LossName: "mse",
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(snap); err != nil {
		t.Fatalf("gob.Encode: %v", err)
	}
	got, err := V0Read(&buf)
	if err != nil {
		t.Fatalf("V0Read: %v", err)
	}
	if got.InputSize != 3 {
		t.Errorf("InputSize=%d, want 3", got.InputSize)
	}
	if len(got.Layers) != 1 {
		t.Fatalf("Layers len=%d, want 1", len(got.Layers))
	}
	if got.Layers[0].WeightShape[0] != 2 || got.Layers[0].WeightShape[1] != 3 {
		t.Errorf("WeightShape=%v, want {2, 3}", got.Layers[0].WeightShape)
	}
}

// TestV1EmptyFileRejected: a 0-byte file is rejected.
func TestV1EmptyFileRejected(t *testing.T) {
	_, err := NewV1Reader(bytes.NewReader(nil))
	if err == nil {
		t.Fatalf("expected error on empty file, got nil")
	}
}

// TestV1TruncatedHeader: a file with fewer than 32 bytes
// is rejected.
func TestV1TruncatedHeader(t *testing.T) {
	_, err := NewV1Reader(bytes.NewReader([]byte{'R', 'E', 'L', 'V', 1}))
	if err == nil {
		t.Fatalf("expected error on truncated header, got nil")
	}
}

// TestV1BF16RoundTrip verifies the bfloat16 wire format
// is bit-exact: write a known set of bf16 values, read
// them back, verify byte-for-byte equality.
func TestV1BF16RoundTrip(t *testing.T) {
	var buf bytes.Buffer
	wr := NewV1Writer(&buf, 0, 4)
	if err := wr.WriteHeader(); err != nil {
		t.Fatalf("WriteHeader: %v", err)
	}
	// Write a few bf16 values.
	in := []uint16{0x3F80, 0x4000, 0x4040, 0x4080} // 1.0, 2.0, 3.0, 4.0
	if err := wr.WriteWeight(in); err != nil {
		t.Fatalf("WriteWeight: %v", err)
	}
	if err := wr.WriteOptimizerState("adam", 0, nil); err != nil {
		t.Fatalf("WriteOptimizerState: %v", err)
	}
	if _, err := wr.WriteFooter(); err != nil {
		t.Fatalf("WriteFooter: %v", err)
	}
	rdr, err := NewV1Reader(&buf)
	if err != nil {
		t.Fatalf("NewV1Reader: %v", err)
	}
	out, err := rdr.ReadWeight()
	if err != nil {
		t.Fatalf("ReadWeight: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("len(out)=%d, want 4", len(out))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("bf16[%d]=%x, want %x", i, out[i], in[i])
		}
	}
	// Sanity: those bit patterns are 1.0, 2.0, 3.0, 4.0.
	if v := transformer.F32FromBF16(out[0]); v != 1.0 {
		t.Errorf("F32FromBF16(0x3F80)=%v, want 1.0", v)
	}
	if v := transformer.F32FromBF16(out[2]); v != 3.0 {
		t.Errorf("F32FromBF16(0x4040)=%v, want 3.0", v)
	}
	// Read the optimizer state to consume the rest of the
	// body before reading the footer.
	if _, _, _, err := rdr.ReadOptimizerState(); err != nil {
		t.Fatalf("ReadOptimizerState: %v", err)
	}
	if err := rdr.ReadFooter(); err != nil {
		t.Fatalf("ReadFooter: %v", err)
	}
}

// contains is a small substring helper that avoids pulling
// in strings just for one test.
func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// errIsEOF: a small helper to compare against io.EOF
// without importing the strings package in the test file
// (errors.Is works fine).
var _ = errors.Is
var _ = io.EOF
