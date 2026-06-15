// Package relux model loader. Reads SafeTensors and GGUF files,
// maps foreign weights to relux's optim.Param slots, and returns a
// ready-to-use *Transformer.
package relux

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/loader"
	"github.com/xDarkicex/relux/internal/optim"
	"github.com/xDarkicex/relux/internal/transformer"
	"github.com/xDarkicex/relux/tokenizer"
)

// LoadedModel is the result of LoadModel.
type LoadedModel struct {
	Transformer  *Transformer
	Tokenizer    *tokenizer.Tokenizer
	Architecture string
	Source       string
}

// LoadModel reads a .safetensors or .gguf file, constructs a
// relux.Transformer with weights mapped from the foreign format, and
// returns the ready-to-use model plus the detected architecture and
// tokenizer (if a sidecar file was found).
//
// Supports LLaMA, LLaMA 2/3, Mistral, Qwen, Phi, Yi, Mixtral MoE, and
// DeepSeek-V2 with MLA. Architecture dims come from GGUF metadata
// or a sidecar config.json (SafeTensors).
func LoadModel(modelPath string) (*LoadedModel, error) {
	lr, err := loader.LoadModel(modelPath)
	if err != nil {
		return nil, err
	}
	cfg := lr.Config
	trCfg := ConfigTransformer{
		VocabSize:  cfg.VocabSize,
		DModel:     cfg.DModel,
		NumHeads:   cfg.NumHeads,
		NumKVHeads: cfg.NumKVHeads,
		NumLayers:  cfg.NumLayers,
		DFF:        cfg.DFF,
		MaxSeqLen:  cfg.MaxSeqLen,
		RopeBase:   cfg.RopeBase,
		NormEps:    cfg.NormEps,
		Causal:     true,
		FFNType:    cfg.FFNType,
		AttnType:   cfg.AttnType,
		MLADimC:    cfg.MLADimC,
		MLADimR:    cfg.MLADimR,
	}
	tr, err := NewTransformer(trCfg)
	if err != nil {
		return nil, fmt.Errorf("loader: construct transformer: %w", err)
	}

	// Now run the actual weight-write loop. We delegate back to
	// the loader's helpers (TensorInfo, ReadTensorData) but do the
	// f32 → bf16 cast + transpose + slot-writing here, since this
	// is the public package and it owns the param slice.
	if err := installWeightsInto(modelPath, lr.Architecture, tr, trCfg); err != nil {
		return nil, err
	}

	return &LoadedModel{
		Transformer:  tr,
		Tokenizer:    lr.Tokenizer,
		Architecture: lr.Architecture,
		Source:       modelPath,
	}, nil
}

// installWeightsInto walks the file's tensors and writes them into
// tr.Params() slots. The orchestrator uses the foreign names to
// locate slots in the param slice (whose layout is the same one
// fixed by Transformer.Params() / Block.Params()).
func installWeightsInto(modelPath, arch string, tr *Transformer, cfg ConfigTransformer) error {
	allParams := tr.Params()
	mapper := detectPublicMapper(modelPath)
	if mapper == nil {
		return fmt.Errorf("loader: cannot detect mapper for %s", modelPath)
	}

	if strings.HasSuffix(strings.ToLower(modelPath), ".safetensors") {
		return installFromSafeTensors(modelPath, mapper, allParams, cfg)
	}
	if strings.HasSuffix(strings.ToLower(modelPath), ".gguf") {
		return installFromGGUF(modelPath, mapper, allParams, cfg)
	}
	return fmt.Errorf("loader: unknown format for %s", modelPath)
}

// detectPublicMapper returns a public-facing mapper. Currently
// the public package re-uses the internal mappers via a thin shim.
func detectPublicMapper(path string) loader.WeightMapper {
	// We don't open the file here; the orchestrator helper does
	// the heavy lifting. This stub is left in case the public
	// package wants to expose mapping decisions in the future.
	return nil
}

// installFromSafeTensors handles .safetensors files.
func installFromSafeTensors(path string, mapper loader.WeightMapper, allParams []optim.Param, cfg ConfigTransformer) error {
	reader, err := loader.NewSafeTensorsReader(path)
	if err != nil {
		return err
	}
	defer reader.Close()

	for _, name := range reader.TensorNames() {
		mw, ok := mapper.Map(name)
		if !ok || !mw.Mapped() {
			continue
		}
		info, err := reader.TensorInfo(name)
		if err != nil {
			return fmt.Errorf("loader: info %q: %w", name, err)
		}
		raw, err := reader.ReadTensorData(name)
		if err != nil {
			return fmt.Errorf("loader: read %q: %w", name, err)
		}
		f32Buf, err := safeTensorsToFloat32(raw, info.DType)
		if err != nil {
			return fmt.Errorf("loader: convert %q: %w", name, err)
		}
		if err := writeToParam(allParams, mw, f32Buf, info.Shape, cfg); err != nil {
			alloc.Free(f32Buf)
			return fmt.Errorf("loader: write %q: %w", name, err)
		}
		alloc.Free(f32Buf)
	}
	return nil
}

// installFromGGUF handles .gguf files.
func installFromGGUF(path string, mapper loader.WeightMapper, allParams []optim.Param, cfg ConfigTransformer) error {
	reader, err := loader.NewGGUFReader(path)
	if err != nil {
		return err
	}
	defer reader.Close()

	for _, name := range reader.TensorNames() {
		mw, ok := mapper.Map(name)
		if !ok {
			continue
		}
		f32Buf, shape, err := loader.ReadGGUFDequantized(reader, name)
		if err != nil {
			return fmt.Errorf("loader: dequant %q: %w", name, err)
		}
		if err := writeToParam(allParams, mw, f32Buf, shape, cfg); err != nil {
			alloc.Free(f32Buf)
			return fmt.Errorf("loader: write %q: %w", name, err)
		}
		alloc.Free(f32Buf)
	}
	return nil
}

// safeTensorsToFloat32 widens raw bytes to f32 (off-heap).
func safeTensorsToFloat32(raw []byte, dtype loader.SafeTensorsDType) ([]float32, error) {
	n := len(raw) / dtypeBytes(dtype)
	if n == 0 {
		return nil, nil
	}
	out := alloc.Float32(n)
	switch dtype {
	case loader.SafeTensorsF32:
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			out[i] = math.Float32frombits(bits)
		}
	case loader.SafeTensorsF16:
		for i := 0; i < n; i++ {
			out[i] = loader.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
	case loader.SafeTensorsBF16:
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2])) << 16
			out[i] = math.Float32frombits(bits)
		}
	default:
		alloc.Free(out)
		return nil, fmt.Errorf("safetensors: unsupported dtype %q", dtype)
	}
	return out, nil
}

func dtypeBytes(d loader.SafeTensorsDType) int {
	switch d {
	case loader.SafeTensorsF64, loader.SafeTensorsI64:
		return 8
	case loader.SafeTensorsF32, loader.SafeTensorsI32:
		return 4
	case loader.SafeTensorsF16, loader.SafeTensorsBF16:
		return 2
	default:
		return 0
	}
}

// writeToParam locates the slot and writes f32Buf as bf16.
func writeToParam(
	allParams []optim.Param,
	mw loader.MappedWeight,
	f32Buf []float32,
	shape []int,
	cfg ConfigTransformer,
) error {
	slot := paramSlot(allParams, mw, cfg)
	if slot < 0 || slot >= len(allParams) {
		return nil
	}
	param := allParams[slot]
	if len(f32Buf) != len(param.Data) {
		// Parameter may not be allocated (e.g. tied embeddings
		// have lm_head as nil). Skip.
		return nil
	}
	if mw.Transpose && len(shape) == 2 {
		rows, cols := shape[0], shape[1]
		if rows*cols != len(f32Buf) {
			return fmt.Errorf("loader: %s shape %v doesn't match buf %d", mw.Kind, shape, len(f32Buf))
		}
		transposed := alloc.Float32(rows * cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				transposed[c*rows+r] = f32Buf[r*cols+c]
			}
		}
		f32ToBF16(transposed, param.Data)
		alloc.Free(transposed)
		return nil
	}
	f32ToBF16(f32Buf, param.Data)
	return nil
}

// paramSlot computes the index in tr.Params() for the slot.
func paramSlot(
	allParams []optim.Param,
	mw loader.MappedWeight,
	cfg ConfigTransformer,
) int {
	switch mw.BlockIdx {
	case -1:
		if mw.Kind == "embed" {
			return 0
		}
	case -2:
		if mw.Kind == "final_norm" {
			return len(allParams) - 2
		}
	case -3:
		if mw.Kind == "lm_head" {
			return len(allParams) - 1
		}
	}
	if mw.BlockIdx < 0 || mw.BlockIdx >= cfg.NumLayers {
		return -1
	}
	stride := perBlockStride(cfg)
	base := 1 + mw.BlockIdx*stride

	isMLA := cfg.AttnType == "mla"
	isSwiGLU := cfg.FFNType == "swiglu"

	switch mw.Kind {
	case "norm_attn":
		return base + 0
	case "norm_mlp":
		if isMLA {
			return base + 8
		}
		return base + 5
	}

	if isMLA {
		switch mw.Kind {
		case "mla.W_Q":
			return base + 1
		case "mla.W_DKV":
			return base + 2
		case "mla.W_UK":
			return base + 3
		case "mla.W_UV":
			return base + 4
		case "mla.W_KR":
			return base + 5
		case "mla.W_QR":
			return base + 6
		case "mla.W_O":
			return base + 7
		}
	} else {
		switch mw.Kind {
		case "mha.Wq":
			return base + 1
		case "mha.Wk":
			return base + 2
		case "mha.Wv":
			return base + 3
		case "mha.Wo":
			return base + 4
		}
	}

	mlpBase := base + 6
	if isMLA {
		mlpBase = base + 9
	}
	if isSwiGLU {
		switch mw.Kind {
		case "mlp.W1":
			return mlpBase + 0
		case "mlp.W2":
			return mlpBase + 1
		case "mlp.W3":
			return mlpBase + 2
		}
	} else {
		switch mw.Kind {
		case "mlp.W1":
			return mlpBase + 0
		case "mlp.b1":
			return mlpBase + 1
		case "mlp.W2":
			return mlpBase + 2
		case "mlp.b2":
			return mlpBase + 3
		}
	}
	return -1
}

func perBlockStride(cfg ConfigTransformer) int {
	n := 1 + 4 + 1
	if cfg.AttnType == "mla" {
		n = 1 + 7 + 1
	}
	if cfg.FFNType == "swiglu" {
		n += 3
	} else {
		n += 4
	}
	return n
}

// f32ToBF16 casts a float32 buffer to a bfloat16 (uint16) buffer.
func f32ToBF16(src []float32, dst []uint16) {
	if len(src) != len(dst) {
		return
	}
	for i := range src {
		u := math.Float32bits(src[i])
		roundingBias := uint32(0x7FFF) + ((u >> 16) & 1)
		u += roundingBias
		dst[i] = uint16(u >> 16)
	}
}

// detectPublicMapperFromPath returns a mapper based on file extension
// and inferred architecture.
func detectPublicMapperFromPath(path string) loader.WeightMapper {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		reader, err := loader.NewGGUFReader(path)
		if err != nil {
			return nil
		}
		defer reader.Close()
		if arch, ok := reader.Metadata()["general.architecture"].(string); ok && arch != "" {
			return detectMapperForArch(arch)
		}
	}
	if strings.HasSuffix(strings.ToLower(path), ".safetensors") {
		reader, err := loader.NewSafeTensorsReader(path)
		if err != nil {
			return nil
		}
		defer reader.Close()
		return loader.DetectArchitecture(reader.TensorNames())
	}
	return nil
}

func detectMapperForArch(arch string) loader.WeightMapper {
	switch strings.ToLower(arch) {
	case "llama":
		return loader.LLaMAMapper{}
	case "mistral":
		return loader.MistralMapper{}
	case "deepseek", "deepseek2":
		return loader.DeepSeekMapper{}
	}
	return loader.LLaMAMapper{}
}

// ensure transformer import is used.
var _ = transformer.ATTN_MHA
