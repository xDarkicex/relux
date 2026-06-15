package loader

import (
	"bytes"
	"encoding/binary"
	"encoding/json"

	"github.com/xDarkicex/relux/internal/alloc"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// --- SafeTensors header parsing ---

// writeSafeTensorsHeader writes a minimal valid SafeTensors file:
// 8-byte header size, JSON header, then 0 bytes of data (the
// tensors themselves aren't actually loaded in these tests — we
// only care about header parsing).
func writeSafeTensorsHeader(t *testing.T, path string, header map[string]interface{}) {
	t.Helper()
	headerJSON, err := jsonMarshal(header)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	buf := &bytes.Buffer{}
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("write header size: %v", err)
	}
	buf.Write(headerJSON)
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write file: %v", err)
	}
}

func TestSafeTensors_Header(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	header := map[string]interface{}{
		"__metadata__": map[string]string{"format": "pt"},
		"model.embed_tokens.weight": map[string]interface{}{
			"dtype":        "BF16",
			"shape":        []int{100, 64},
			"data_offsets": []int{0, 12800},
		},
		"model.layers.0.self_attn.q_proj.weight": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{64, 64},
			"data_offsets": []int{12800, 29184},
		},
		"model.layers.1.self_attn.k_proj.weight": map[string]interface{}{
			"dtype":        "F16",
			"shape":        []int{64, 32},
			"data_offsets": []int{29184, 37376},
		},
	}
	writeSafeTensorsHeader(t, path, header)

	r, err := NewSafeTensorsReader(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	names := r.TensorNames()
	if len(names) != 3 {
		t.Errorf("expected 3 tensors (excl. metadata), got %d: %v", len(names), names)
	}

	info, err := r.TensorInfo("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("info: %v", err)
	}
	if info.DType != SafeTensorsBF16 {
		t.Errorf("dtype: want BF16, got %q", info.DType)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 100 || info.Shape[1] != 64 {
		t.Errorf("shape: want [100 64], got %v", info.Shape)
	}
	if info.NumBytes() != 12800 {
		t.Errorf("bytes: want 12800, got %d", info.NumBytes())
	}
	if info.NumElements() != 6400 {
		t.Errorf("elements: want 6400, got %d", info.NumElements())
	}
}

// --- Dequantizer unit tests ---

func TestDequantize_F32(t *testing.T) {
	src := []float32{1.5, -2.25, 100.0, 0.0}
	buf := &bytes.Buffer{}
	for _, v := range src {
		binary.Write(buf, binary.LittleEndian, math.Float32bits(v))
	}
	out := alloc.Float32(4)
	n, err := Dequantize(buf.Bytes(), GGMLTypeF32, 4, out)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if n != 4 {
		t.Errorf("n: want 4, got %d", n)
	}
	for i, v := range src {
		if out[i] != v {
			t.Errorf("[%d]: want %v, got %v", i, v, out[i])
		}
	}
}

func TestDequantize_F16(t *testing.T) {
	// 0x3C00 = 1.0, 0x4000 = 2.0, 0xBC00 = -1.0 in IEEE 754 half.
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x3C00, 1.0},
		{0x4000, 2.0},
		{0xBC00, -1.0},
		{0x0000, 0.0},
		{0x7C00, float32(math.Inf(1))},
	}
	buf := &bytes.Buffer{}
	for _, c := range cases {
		binary.Write(buf, binary.LittleEndian, c.bits)
	}
	out := alloc.Float32(len(cases))
	if _, err := Dequantize(buf.Bytes(), GGMLTypeF16, len(cases), out); err != nil {
		t.Fatalf("dequant: %v", err)
	}
	for i, c := range cases {
		if math.IsInf(float64(c.want), 0) {
			if !math.IsInf(float64(out[i]), 0) {
				t.Errorf("[%d]: want Inf, got %v", i, out[i])
			}
		} else if out[i] != c.want {
			t.Errorf("[%d] (bits=0x%04X): want %v, got %v", i, c.bits, c.want, out[i])
		}
	}
}

func TestDequantize_BF16(t *testing.T) {
	// 0x3F80 = 1.0, 0x4000 = 2.0, 0xBF80 = -1.0 in bf16.
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x3F80, 1.0},
		{0x4000, 2.0},
		{0xBF80, -1.0},
	}
	buf := &bytes.Buffer{}
	for _, c := range cases {
		binary.Write(buf, binary.LittleEndian, c.bits)
	}
	out := alloc.Float32(len(cases))
	if _, err := Dequantize(buf.Bytes(), GGMLTypeBF16, len(cases), out); err != nil {
		t.Fatalf("dequant: %v", err)
	}
	for i, c := range cases {
		if out[i] != c.want {
			t.Errorf("[%d] (bits=0x%04X): want %v, got %v", i, c.bits, c.want, out[i])
		}
	}
}

func TestDequantize_Q8_0(t *testing.T) {
	// One Q8_0 block: 32 elements. 2 bytes f16 scale + 32 int8 values.
	// scale=1.0, values=0..31 → output 0.0..31.0.
	buf := &bytes.Buffer{}
	binary.Write(buf, binary.LittleEndian, uint16(0x3C00)) // f16 1.0: sign=0, exp=15, man=0
	for i := int8(0); i < 32; i++ {
		buf.WriteByte(byte(i))
	}
	out := alloc.Float32(32)
	if _, err := Dequantize(buf.Bytes(), GGMLTypeQ8_0, 32, out); err != nil {
		t.Fatalf("dequant: %v", err)
	}
	for i := 0; i < 32; i++ {
		if math.Abs(float64(out[i]-float32(i))) > 0.01 {
			t.Errorf("[%d]: want ~%d, got %v", i, i, out[i])
		}
	}
}

// --- Weight mapper unit tests ---

func TestLLaMAMapper(t *testing.T) {
	m := LLaMAMapper{}
	cases := []struct {
		name    string
		wantMW  MappedWeight
		wantOk  bool
	}{
		{"model.embed_tokens.weight", MappedWeight{BlockIdx: -1, Kind: "embed"}, true},
		{"model.norm.weight", MappedWeight{BlockIdx: -2, Kind: "final_norm"}, true},
		{"lm_head.weight", MappedWeight{BlockIdx: -3, Kind: "lm_head", Transpose: true}, true},
		{"model.layers.0.self_attn.q_proj.weight", MappedWeight{BlockIdx: 0, Kind: "mha.Wq", Transpose: true}, true},
		{"model.layers.5.self_attn.k_proj.weight", MappedWeight{BlockIdx: 5, Kind: "mha.Wk", Transpose: true}, true},
		{"model.layers.7.self_attn.v_proj.weight", MappedWeight{BlockIdx: 7, Kind: "mha.Wv", Transpose: true}, true},
		{"model.layers.3.self_attn.o_proj.weight", MappedWeight{BlockIdx: 3, Kind: "mha.Wo", Transpose: true}, true},
		{"model.layers.0.mlp.gate_proj.weight", MappedWeight{BlockIdx: 0, Kind: "mlp.W1", Transpose: true}, true},
		{"model.layers.0.mlp.up_proj.weight", MappedWeight{BlockIdx: 0, Kind: "mlp.W3", Transpose: true}, true},
		{"model.layers.0.mlp.down_proj.weight", MappedWeight{BlockIdx: 0, Kind: "mlp.W2", Transpose: true}, true},
		{"model.layers.0.input_layernorm.weight", MappedWeight{BlockIdx: 0, Kind: "norm_attn"}, true},
		{"model.layers.0.post_attention_layernorm.weight", MappedWeight{BlockIdx: 0, Kind: "norm_mlp"}, true},
		{"some.unrelated.weight", MappedWeight{}, false},
	}
	for _, c := range cases {
		got, ok := m.Map(c.name)
		if ok != c.wantOk {
			t.Errorf("%s: ok want %v, got %v", c.name, c.wantOk, ok)
			continue
		}
		if !ok {
			continue
		}
		if got.BlockIdx != c.wantMW.BlockIdx || got.Kind != c.wantMW.Kind || got.Transpose != c.wantMW.Transpose {
			t.Errorf("%s: want %+v, got %+v", c.name, c.wantMW, got)
		}
	}
	if m.Architecture() != "llama" {
		t.Errorf("architecture: want llama, got %q", m.Architecture())
	}
}

func TestDeepSeekMapper(t *testing.T) {
	m := DeepSeekMapper{}
	cases := []struct {
		name   string
		kind   string
		idx    int
	}{
		{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "mla.W_DKV", 0},
		{"model.layers.5.self_attn.kv_b_proj.weight", "mla.W_UK_W_UV", 5},
		{"model.layers.3.self_attn.k_pe_proj.weight", "mla.W_KR", 3},
		{"model.layers.0.self_attn.q_proj.weight", "mha.Wq", 0}, // LLaMA fallback
		{"model.layers.0.mlp.gate_proj.weight", "mlp.W1", 0},
	}
	for _, c := range cases {
		got, ok := m.Map(c.name)
		if !ok {
			t.Errorf("%s: should map", c.name)
			continue
		}
		if got.Kind != c.kind || got.BlockIdx != c.idx {
			t.Errorf("%s: want {%d %q}, got %+v", c.name, c.idx, c.kind, got)
		}
	}
	if m.Architecture() != "deepseek" {
		t.Errorf("architecture: want deepseek, got %q", m.Architecture())
	}
}

func TestDetectArchitecture(t *testing.T) {
	// DeepSeek detection.
	ds := []string{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.norm.weight"}
	if got := DetectArchitecture(ds).Architecture(); got != "deepseek" {
		t.Errorf("deepseek detection: got %q", got)
	}
	// Mixtral detection.
	mix := []string{"model.layers.0.block_sparse_moe.experts.0.w1.weight", "model.norm.weight"}
	if got := DetectArchitecture(mix).Architecture(); got != "mistral" {
		t.Errorf("mistral detection: got %q", got)
	}
	// LLaMA default.
	ll := []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"}
	if got := DetectArchitecture(ll).Architecture(); got != "llama" {
		t.Errorf("llama detection: got %q", got)
	}
}

// --- GGUF header parsing ---

func TestGGUF_Header(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.gguf")

	// Build a minimal GGUF v3 file by hand.
	buf := &bytes.Buffer{}
	// Magic + version
	binary.Write(buf, binary.LittleEndian, uint32(ggufMagicLE))
	binary.Write(buf, binary.LittleEndian, uint32(ggufVersionV3))
	binary.Write(buf, binary.LittleEndian, uint64(2)) // 2 tensors
	binary.Write(buf, binary.LittleEndian, uint64(1)) // 1 metadata KV

	// Metadata KV 1: key="general.architecture", value=string "llama"
	writeStringField(buf, "general.architecture")
	writeValueType(buf, GGUFValueString)
	writeStringField(buf, "llama")

	// Tensor 1: "test.weight", shape=[2,2], F32, offset=0, 16 bytes data
	writeStringField(buf, "test.weight")
	binary.Write(buf, binary.LittleEndian, uint32(2))  // ndims
	binary.Write(buf, binary.LittleEndian, uint64(2))
	binary.Write(buf, binary.LittleEndian, uint64(2))
	binary.Write(buf, binary.LittleEndian, uint32(GGMLTypeF32))
	binary.Write(buf, binary.LittleEndian, uint64(0)) // offset

	// Tensor 2: "other.bias", shape=[4], F32, offset=16
	writeStringField(buf, "other.bias")
	binary.Write(buf, binary.LittleEndian, uint32(1)) // ndims
	binary.Write(buf, binary.LittleEndian, uint64(4))
	binary.Write(buf, binary.LittleEndian, uint32(GGMLTypeF32))
	binary.Write(buf, binary.LittleEndian, uint64(16)) // offset

	// Pad to alignment, then data.
	pos := buf.Len()
	aligned := alignUpInt(pos, ggufDefaultAlignment)
	for i := 0; i < aligned-pos; i++ {
		buf.WriteByte(0)
	}
	// Write 32 bytes of data: 2x2 f32 + 4 f32
	for i := 0; i < 8; i++ {
		binary.Write(buf, binary.LittleEndian, float32(i+1))
	}

	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	r, err := NewGGUFReader(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer r.Close()

	if arch, ok := r.Metadata()["general.architecture"].(string); !ok || arch != "llama" {
		t.Errorf("metadata arch: want llama, got %v", r.Metadata()["general.architecture"])
	}
	if len(r.TensorNames()) != 2 {
		t.Errorf("tensor count: want 2, got %d", len(r.TensorNames()))
	}

	t1, err := r.TensorInfo("test.weight")
	if err != nil {
		t.Fatalf("info test.weight: %v", err)
	}
	if t1.DType != GGMLTypeF32 || t1.NumElement != 4 {
		t.Errorf("test.weight: dtype=%v nelem=%d", t1.DType, t1.NumElement)
	}

	// Read tensor data (zero-copy into mmap).
	raw, err := r.ReadTensorData("test.weight")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(raw) != 16 {
		t.Errorf("test.weight bytes: want 16, got %d", len(raw))
	}
}

func writeStringField(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

func writeValueType(buf *bytes.Buffer, t GGUFValueType) {
	binary.Write(buf, binary.LittleEndian, uint32(t))
}

func alignUpInt(v, align int) int {
	if align <= 0 {
		return v
	}
	return (v + align - 1) & ^(align - 1)
}

// helper: minimal JSON marshal using stdlib.
func jsonMarshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}
