package loader

import (
	"fmt"
	"strconv"
	"strings"
)

// Architecture identifiers.
const (
	ArchitectureLLaMA    = "llama"
	ArchitectureMistral  = "mistral"
	ArchitectureDeepSeek = "deepseek"
)

// MappedWeight describes where a foreign tensor name should land in
// relux's parameter graph.
//
// BlockIdx is the transformer block index (0..NumLayers-1). For
// non-block tensors (embedding, final norm, lm_head) it carries
// sentinel negative values:
//
//	-1 = embedding
//	-2 = final norm
//	-3 = lm_head
//
// Kind is the relux param slot identifier. The string maps directly
// to a slot in the params slice returned by Transformer.Params():
//
//	"embed"      →  embedding weight
//	"norm_attn"  →  block{i}.normAttn.gamma
//	"norm_mlp"   →  block{i}.normMlp.gamma
//	"mha.Wq"     →  block{i}.mha.Wq
//	"mha.Wk"     →  block{i}.mha.Wk
//	"mha.Wv"     →  block{i}.mha.Wv
//	"mha.Wo"     →  block{i}.mha.Wo
//	"mlp.W1"     →  block{i}.mlp.W1   (gate in LLaMA naming)
//	"mlp.W2"     →  block{i}.mlp.W2   (down)
//	"mlp.W3"     →  block{i}.mlp.WGate (up in LLaMA naming)
//	"final_norm" →  final norm gamma
//	"lm_head"    →  lm_head weight
//
// Transpose: PyTorch stores linear weights as (out, in). relux's
// matmulBatched3D expects (in, out). When true, the loader must
// transpose the loaded matrix before writing it to param.Data.
type MappedWeight struct {
	BlockIdx  int
	Kind      string
	Transpose bool
}

// Mapped reports whether Map() recognized the tensor. Names that
// aren't part of the model's parameter set (e.g. tokenizer tensors)
// are unmapped and skipped.
func (m MappedWeight) Mapped() bool { return m.Kind != "" }

// WeightMapper translates foreign tensor names to relux slots.
type WeightMapper interface {
	Map(name string) (MappedWeight, bool)
	Architecture() string
}

// LLaMAMapper handles LLaMA, LLaMA 2, LLaMA 3, Mistral, Qwen, Phi, Yi.
// All use the same HuggingFace naming convention.
type LLaMAMapper struct{}

// Map translates a LLaMA-family tensor name.
func (LLaMAMapper) Map(name string) (MappedWeight, bool) {
	// Embedding.
	if name == "model.embed_tokens.weight" {
		return MappedWeight{BlockIdx: -1, Kind: "embed"}, true
	}
	// Final norm.
	if name == "model.norm.weight" {
		return MappedWeight{BlockIdx: -2, Kind: "final_norm"}, true
	}
	// LM head.
	if name == "lm_head.weight" {
		return MappedWeight{BlockIdx: -3, Kind: "lm_head", Transpose: true}, true
	}

	// Block-prefixed tensors.
	const prefix = "model.layers."
	if !strings.HasPrefix(name, prefix) {
		return MappedWeight{}, false
	}
	rest := name[len(prefix):]
	dot := strings.IndexByte(rest, '.')
	if dot < 0 {
		return MappedWeight{}, false
	}
	idx, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return MappedWeight{}, false
	}

	switch {
	case strings.HasSuffix(name, ".self_attn.q_proj.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "mha.Wq", Transpose: true}, true
	case strings.HasSuffix(name, ".self_attn.k_proj.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "mha.Wk", Transpose: true}, true
	case strings.HasSuffix(name, ".self_attn.v_proj.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "mha.Wv", Transpose: true}, true
	case strings.HasSuffix(name, ".self_attn.o_proj.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "mha.Wo", Transpose: true}, true
	case strings.HasSuffix(name, ".mlp.gate_proj.weight"):
		// LLaMA naming: gate_proj = W1 (the "up" / "in" half of SwiGLU).
		return MappedWeight{BlockIdx: idx, Kind: "mlp.W1", Transpose: true}, true
	case strings.HasSuffix(name, ".mlp.up_proj.weight"):
		// LLaMA naming: up_proj = W3 (the "gate" of SwiGLU).
		return MappedWeight{BlockIdx: idx, Kind: "mlp.W3", Transpose: true}, true
	case strings.HasSuffix(name, ".mlp.down_proj.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "mlp.W2", Transpose: true}, true
	case strings.HasSuffix(name, ".input_layernorm.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "norm_attn"}, true
	case strings.HasSuffix(name, ".post_attention_layernorm.weight"):
		return MappedWeight{BlockIdx: idx, Kind: "norm_mlp"}, true
	}
	return MappedWeight{}, false
}

// Architecture returns "llama".
func (LLaMAMapper) Architecture() string { return ArchitectureLLaMA }

// MistralMapper is the same as LLaMAMapper for now (Mistral uses
// identical weight naming). Kept as a separate type so future Mistral-
// specific weight names (e.g. sliding-window attention flags) can
// be handled without affecting LLaMA.
type MistralMapper struct{}

// Map delegates to LLaMAMapper.
func (MistralMapper) Map(name string) (MappedWeight, bool) {
	return LLaMAMapper{}.Map(name)
}

// Architecture returns "mistral".
func (MistralMapper) Architecture() string { return ArchitectureMistral }

// DeepSeekMapper handles DeepSeek-V1/V2. Non-MLA weights follow
// LLaMA naming; MLA-specific weights map to:
//   - self_attn.kv_a_proj_with_mqa.weight → block{i}.mla.W_DKV  (latent down-projection)
//   - self_attn.kv_b_proj.weight           → block{i}.mla.W_UK + W_UV (split via shape)
//   - self_attn.k_pe_proj.weight           → block{i}.mla.W_KR  (decoupled RoPE key)
//   - self_attn.q_pe_proj.weight           → block{i}.mla.W_QR  (decoupled RoPE query, if present)
//
// In DeepSeek-V2 the kv_b_proj tensor is conceptually two matrices
// (W_UK and W_UV) stored as a single fused weight. The Shape field
// of the foreign tensor distinguishes them: if the second-to-last
// dim is numHeads*headDim, the row count is numHeads*(headDim + dC),
// and the tensor splits into W_UK (first dC cols) and W_UV (last
// headDim cols). The loader handles this split during weight writing.
type DeepSeekMapper struct{}

// Map translates a DeepSeek tensor name.
func (m DeepSeekMapper) Map(name string) (MappedWeight, bool) {
	// MLA-specific keys.
	switch {
	case strings.HasSuffix(name, ".self_attn.kv_a_proj_with_mqa.weight"):
		idx, ok := extractDeepSeekBlockIdx(name)
		if !ok {
			return MappedWeight{}, false
		}
		return MappedWeight{BlockIdx: idx, Kind: "mla.W_DKV", Transpose: true}, true
	case strings.HasSuffix(name, ".self_attn.kv_b_proj.weight"):
		// Caller splits into W_UK + W_UV by reading the tensor
		// name twice and inspecting shape. We mark the kind
		// ambiguously; the loader uses the global metadata
		// (numHeads, headDim, dC) to route to the right slot.
		idx, ok := extractDeepSeekBlockIdx(name)
		if !ok {
			return MappedWeight{}, false
		}
		return MappedWeight{BlockIdx: idx, Kind: "mla.W_UK_W_UV", Transpose: true}, true
	case strings.HasSuffix(name, ".self_attn.k_pe_proj.weight"):
		idx, ok := extractDeepSeekBlockIdx(name)
		if !ok {
			return MappedWeight{}, false
		}
		return MappedWeight{BlockIdx: idx, Kind: "mla.W_KR", Transpose: true}, true
	}

	// Everything else follows LLaMA naming.
	return LLaMAMapper{}.Map(name)
}

// Architecture returns "deepseek".
func (DeepSeekMapper) Architecture() string { return ArchitectureDeepSeek }

// extractDeepSeekBlockIdx pulls the layer index from a "model.layers.{i}..." path.
func extractDeepSeekBlockIdx(name string) (int, bool) {
	const prefix = "model.layers."
	if !strings.HasPrefix(name, prefix) {
		return 0, false
	}
	rest := name[len(prefix):]
	dot := strings.IndexByte(rest, '.')
	if dot < 0 {
		return 0, false
	}
	idx, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return 0, false
	}
	return idx, true
}

// DetectArchitecture picks a WeightMapper from the model's tensor
// names. Heuristics, in order:
//
//  1. "kv_a_proj" or "kv_b_proj" present → DeepSeek
//  2. "block_sparse_moe" present → Mistral (Mixtral MoE)
//  3. default → LLaMA
func DetectArchitecture(tensorNames []string) WeightMapper {
	for _, n := range tensorNames {
		if strings.Contains(n, "kv_a_proj") || strings.Contains(n, "kv_b_proj") {
			return DeepSeekMapper{}
		}
	}
	for _, n := range tensorNames {
		if strings.Contains(n, "block_sparse_moe") {
			return MistralMapper{}
		}
	}
	return LLaMAMapper{}
}

// String returns a one-line description of the mapper for logging.
func String(m WeightMapper) string {
	return fmt.Sprintf("WeightMapper(%s)", m.Architecture())
}
