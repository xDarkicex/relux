// Package loader parses SafeTensors and GGUF model files and exposes
// a pure data-loading API. The orchestration that turns the
// loaded params into a live *relux.Transformer lives in the public
// relux package (loader.go) to avoid an import cycle.
package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/optim"
	"github.com/xDarkicex/relux/tokenizer"
)

// MappedWeight is re-exported from mapper.go for convenience.
// The mapper also owns the Mapped/WeightMapper types.

// TensorInfo is the metadata one loader hands to the orchestrator.
type TensorInfo struct {
	DType  string
	Shape  []int
}

// ModelConfig is the architecture dims the loader derived.
type ModelConfig struct {
	VocabSize       int
	DModel          int
	NumHeads        int
	NumKVHeads      int
	NumLayers       int
	DFF             int
	MaxSeqLen       int
	RopeBase        float32
	NormEps         float32
	FFNType         string
	AttnType        string
	MLADimC         int
	MLADimR         int
}

// LoadResult is the return value of the pure data-loading API.
type LoadResult struct {
	Params        []optim.Param // bf16 weights, ready to be installed into a Transformer
	Config        ModelConfig
	Architecture  string
	Tokenizer     *tokenizer.Tokenizer
	Source        string
}

// LoadModel reads a .safetensors or .gguf file and returns a
// LoadResult with the architecture config, all params, and the
// tokenizer (if a sidecar file was found).
func LoadModel(modelPath string) (*LoadResult, error) {
	ext := strings.ToLower(filepath.Ext(modelPath))
	switch ext {
	case ".safetensors":
		return loadSafeTensorsModel(modelPath)
	case ".gguf":
		return loadGGUFModel(modelPath)
	default:
		return nil, fmt.Errorf("loader: unsupported extension %q", ext)
	}
}

func loadSafeTensorsModel(path string) (*LoadResult, error) {
	reader, err := NewSafeTensorsReader(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	names := reader.TensorNames()
	mapper := DetectArchitecture(names)

	cfg, err := configFromSafeTensors(path, names, reader)
	if err != nil {
		return nil, err
	}

	// Construct a fake param slice of the right size by calling
	// the public relux package's transformer constructors.
	// The public package's loader.go does this — we just hand
	// back the config here.
	result := &LoadResult{
		Config:       *cfg,
		Architecture: mapper.Architecture(),
		Source:       path,
	}
	if tok, err := loadSidecarTokenizer(path); err == nil {
		result.Tokenizer = tok
	}
	return result, nil
}

func loadGGUFModel(path string) (*LoadResult, error) {
	reader, err := NewGGUFReader(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	names := reader.TensorNames()
	mapper := detectFromGGUFMetadata(reader, names)

	cfg, err := configFromGGUF(reader)
	if err != nil {
		return nil, err
	}

	result := &LoadResult{
		Config:       *cfg,
		Architecture: mapper.Architecture(),
		Source:       path,
	}
	if tok, err := loadSidecarTokenizer(path); err == nil {
		result.Tokenizer = tok
	}
	return result, nil
}

// ReadTensorDequantized reads a tensor from a SafeTensors reader
// and widens it to a freshly allocated []float32 (off-heap).
func ReadTensorDequantized(r *SafeTensorsReader, name string) ([]float32, []int, error) {
	info, err := r.TensorInfo(name)
	if err != nil {
		return nil, nil, err
	}
	raw, err := r.ReadTensorData(name)
	if err != nil {
		return nil, nil, err
	}
	f32, err := safeTensorsToFloat32(raw, info.DType)
	if err != nil {
		return nil, nil, err
	}
	return f32, info.Shape, nil
}

// ReadGGUFDequantized reads a tensor from a GGUF reader,
// dequantizes, and returns the f32 data and row-major shape.
func ReadGGUFDequantized(r *GGUFReader, name string) ([]float32, []int, error) {
	tinfo, err := r.TensorInfo(name)
	if err != nil {
		return nil, nil, err
	}
	raw, err := r.ReadTensorData(name)
	if err != nil {
		return nil, nil, err
	}
	out, err := allocFloat32(int(tinfo.NumElement))
	if err != nil {
		return nil, nil, err
	}
	if _, err := Dequantize(raw, tinfo.DType, int(tinfo.NumElement), out); err != nil {
		return nil, nil, err
	}
	return out, ggufShapeToRowMajor(tinfo.Dims), nil
}

// ggufShapeToRowMajor reverses GGUF dims into row-major (Go int).
func ggufShapeToRowMajor(dims []uint64) []int {
	// []int is 8 bytes on 64-bit — same as int64. Use the typed
	// int64 allocator and re-cast.
	i64 := alloc.Int64(len(dims))
	for i, d := range dims {
		i64[len(dims)-1-i] = int64(d) //nolint:gosec
	}
	// unsafe-cast: the underlying storage is the same layout.
	return *(*[]int)(unsafe.Pointer(&i64))
}

// safeTensorsToFloat32 widens raw bytes to f32.
func safeTensorsToFloat32(raw []byte, dtype SafeTensorsDType) ([]float32, error) {
	n := len(raw) / dtypeBytes(dtype)
	if n == 0 {
		return nil, nil
	}
	out, err := allocFloat32(n)
	if err != nil {
		return nil, err
	}
	switch dtype {
	case SafeTensorsF32:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			out[i] = math.Float32frombits(bits)
		}
	case SafeTensorsF16:
		for i := 0; i < n; i++ {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
	case SafeTensorsBF16:
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2])) << 16
			out[i] = math.Float32frombits(bits)
		}
	case SafeTensorsF64:
		for i := 0; i < n; i++ {
			out[i] = float32(binary.LittleEndian.Uint64(raw[i*8 : i*8+8]))
		}
	default:
		alloc.Free(out)
		return nil, fmt.Errorf("safetensors: unsupported dtype %q", dtype)
	}
	return out, nil
}

func dtypeBytes(d SafeTensorsDType) int {
	switch d {
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

// allocFloat32 is the off-heap f32 allocator. Wired in this file's
// init() so gguf.go can call it without depending on alloc.
var allocFloat32 func(n int) ([]float32, error)

func init() {
	allocFloat32 = func(n int) ([]float32, error) {
		if n == 0 {
			return nil, nil
		}
		return alloc.Float32(n), nil
	}
}

// configFromSafeTensors reads config.json sidecar or derives dims.
func configFromSafeTensors(path string, names []string, reader *SafeTensorsReader) (*ModelConfig, error) {
	dir := filepath.Dir(path)
	sidecar := filepath.Join(dir, "config.json")
	if data, err := os.ReadFile(sidecar); err == nil {
		raw := map[string]interface{}{}
		if err := json.Unmarshal(data, &raw); err == nil {
			cfg := parseHFConfig(raw)
			return &cfg, nil
		}
	}
	numLayers := countLayerPrefix(names, "model.layers.")
	if numLayers == 0 {
		return nil, fmt.Errorf("loader: cannot derive architecture from %s", path)
	}
	vocab, dModel := 32000, 4096
	for _, n := range names {
		if n == "model.embed_tokens.weight" {
			info, err := reader.TensorInfo(n)
			if err == nil && len(info.Shape) >= 2 {
				vocab = info.Shape[0]
				dModel = info.Shape[1]
			}
		}
	}
	return &ModelConfig{
		VocabSize:  vocab,
		DModel:     dModel,
		NumHeads:   8,
		NumKVHeads: 8,
		NumLayers:  numLayers,
		DFF:        4 * dModel,
		MaxSeqLen:  2048,
		RopeBase:   10000,
		NormEps:    1e-5,
	}, nil
}

// configFromGGUF reads dims from metadata.
func configFromGGUF(r *GGUFReader) (*ModelConfig, error) {
	meta := r.Metadata()
	get := func(key string, def int) int {
		if v, ok := meta[key]; ok {
			switch x := v.(type) {
			case int32:
				return int(x)
			case uint32:
				return int(x)
			case uint64:
				return int(x)
			case int64:
				return int(x)
			case float32:
				return int(x)
			}
		}
		return def
	}
	arch, _ := meta["general.architecture"].(string)
	dModel := get(arch+".embedding_length", 4096)
	numHeads := get(arch+".attention.head_count", 32)
	numKVHeads := get(arch+".attention.head_count_kv", numHeads)
	numLayers := get(arch+".block_count", 32)
	dFF := get(arch+".feed_forward_length", 4*dModel)
	vocab := 0
	if t, ok := r.tensors["token_embd.weight"]; ok && len(t.Dims) > 0 {
		vocab = int(t.Dims[len(t.Dims)-1]) //nolint:gosec
	}
	maxSeq := get(arch+".context_length", 2048)
	ropeBase := float32(get(arch+".rope.freq_base", 10000))

	attnType := "mha"
	var mlaDimC, mlaDimR int
	if strings.Contains(strings.ToLower(arch), "deepseek") {
		attnType = "mla"
		mlaDimC = get(arch+".attention.kv_lora_rank", 0)
		if mlaDimC == 0 {
			mlaDimC = 4 * (dModel / numHeads)
		}
		mlaDimR = get(arch+".attention.nope_head_dim", 0)
		if mlaDimR == 0 {
			mlaDimR = (dModel / numHeads) / 2
		}
	}

	return &ModelConfig{
		VocabSize:  vocab,
		DModel:     dModel,
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		NumLayers:  numLayers,
		DFF:        dFF,
		MaxSeqLen:  maxSeq,
		RopeBase:   ropeBase,
		NormEps:    1e-5,
		AttnType:   attnType,
		MLADimC:    mlaDimC,
		MLADimR:    mlaDimR,
	}, nil
}

// parseHFConfig maps a HuggingFace config.json to ModelConfig.
func parseHFConfig(raw map[string]interface{}) ModelConfig {
	hfGet := func(key string) float64 {
		if v, ok := raw[key]; ok {
			if f, ok := v.(float64); ok {
				return f
			}
			if i, ok := v.(int); ok {
				return float64(i)
			}
		}
		return 0
	}
	hfGetStr := func(key string) string {
		if v, ok := raw[key].(string); ok {
			return v
		}
		return ""
	}

	cfg := ModelConfig{
		DModel:     int(hfGet("hidden_size")),
		NumHeads:   int(hfGet("num_attention_heads")),
		NumKVHeads: int(hfGet("num_key_value_heads")),
		NumLayers:  int(hfGet("num_hidden_layers")),
		DFF:        int(hfGet("intermediate_size")),
		VocabSize:  int(hfGet("vocab_size")),
		MaxSeqLen:  int(hfGet("max_position_embeddings")),
		RopeBase:   float32(hfGet("rope_theta")),
		NormEps:    float32(hfGet("rms_norm_eps")),
	}
	if cfg.NumKVHeads == 0 {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if cfg.MaxSeqLen == 0 {
		cfg.MaxSeqLen = int(hfGet("max_sequence_length"))
	}
	if cfg.RopeBase == 0 {
		cfg.RopeBase = 10000
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-5
	}
	act := hfGetStr("hidden_act")
	if act == "silu" {
		cfg.FFNType = "swiglu"
	}
	if arch := hfGetStr("model_type"); strings.Contains(strings.ToLower(arch), "deepseek") {
		cfg.AttnType = "mla"
		if kv, ok := raw["kv_lora_rank"].(float64); ok {
			cfg.MLADimC = int(kv)
		}
		if nhd, ok := raw["nope_head_dim"].(float64); ok {
			cfg.MLADimR = int(nhd)
		}
	}
	return cfg
}

func countLayerPrefix(names []string, prefix string) int {
	maxIdx := -1
	for _, n := range names {
		if !strings.HasPrefix(n, prefix) {
			continue
		}
		rest := n[len(prefix):]
		dot := strings.IndexByte(rest, '.')
		if dot < 0 {
			continue
		}
		var idx int
		if _, err := fmt.Sscanf(rest[:dot], "%d", &idx); err == nil {
			if idx > maxIdx {
				maxIdx = idx
			}
		}
	}
	return maxIdx + 1
}

func loadSidecarTokenizer(modelPath string) (*tokenizer.Tokenizer, error) {
	dir := filepath.Dir(modelPath)
	candidates := []string{
		filepath.Join(dir, "tokenizer.json"),
		filepath.Join(filepath.Dir(dir), "tokenizer.json"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return tokenizer.Load(p)
		}
	}
	return nil, fmt.Errorf("no tokenizer.json found")
}

// detectFromGGUFMetadata picks a WeightMapper from the GGUF
// "general.architecture" metadata key, falling back to name heuristics.
func detectFromGGUFMetadata(r *GGUFReader, names []string) WeightMapper {
	if arch, ok := r.Metadata()["general.architecture"].(string); ok && arch != "" {
		switch strings.ToLower(arch) {
		case "llama":
			return LLaMAMapper{}
		case "mistral":
			return MistralMapper{}
		case "deepseek", "deepseek2":
			return DeepSeekMapper{}
		}
	}
	return DetectArchitecture(names)
}

// allocFloat32 is wired to alloc.Float32 in mapper.go's init.
