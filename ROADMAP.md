# relux — Roadmap

> From a Go MLP framework to a full mixed-precision transformer runtime.
> This document tracks the direction of the project. Items marked **✅** have
> shipped in `master` and are exercised by the test suite. **🔄** items are
> in flight. **🔮** items are scoped but not yet started.

---

## Vision

relux is a Go-first machine learning framework that runs anywhere pure Go
runs, with optional hardware acceleration through the companion
[**rnxa**](https://github.com/xDarkicex/rnxa) engine. The long-term
direction is a from-scratch implementation of modern transformer LLM
training and inference — embeddings, attention, MLP, RoPE, RMSNorm,
mixed precision, Adam — running natively on Apple Silicon through MPS /
Metal, with CUDA and CPU fallbacks.

The MLP path (`relux.Network`) is the legacy entry point. The new
`relux.Transformer` is the focus of new feature work.

---

## Foundation — Shipped ✅

### Core layers
| Area | Notes |
|------|-------|
| Dense (fully-connected) | `internal/layer/dense.go` — bf16 weights, f32 grads |
| Embedding | `internal/transformer/embedding.go` — bf16 weight table, f32 lookup output |
| RMSNorm | `internal/transformer/rmsnorm.go` — pre-norm, f32 accumulation |
| LayerNorm | `internal/transformer/layernorm.go` — pre-norm, f32 accumulation |
| Multi-Head Attention | `internal/transformer/mha.go` — GQA + RoPE + causal mask, all-bf16 weight path |
| RoPE | `internal/transformer/rope.go` — precomputed cos/sin tables, headDim must be even |
| MLP (FFN block) | `internal/transformer/mlp.go` — GELU exact, 2-layer |
| Linear (head) | `internal/transformer/linear.go` — single y = x @ W + b |
| Transformer Block | `internal/transformer/transformer_block.go` — pre-norm composition |
| Activation suite | ReLU, Sigmoid, Tanh, Softmax, Identity, LeakyReLU, GELU, Swish |
| Loss suite | MSE, BCE, Categorical CE, Sparse CE |

### Optimizers
| Area | Notes |
|------|-------|
| SGD with momentum | `internal/optim/sgd.go` — float32 state |
| Adam | `internal/optim/adam.go` — float32 m/v (NOT bf16 — see Mixed Precision) |
| Gradient clipping | `optim.ClipGradNorm` — float32 in / out |

### Mixed precision (bf16 execution format)
| Area | Notes |
|------|-------|
| Weights stored as bf16 | `optim.Param.Data []uint16` (the bit pattern) |
| Gradients in float32 | `optim.Param.Grad []float32` — standard mixed-precision practice |
| Adam m/v in float32 | preserves the precision of running averages with β₂=0.999 |
| bf16 matmul helpers | `internal/transformer/mha.go::matmulBatched3D` widens bf16 → f32 on the fly |
| Network (MLP) and Transformer both use bf16 | the entire codebase is on the same numeric contract |

### Training
| Area | Notes |
|------|-------|
| LR decay, early stopping, gradient clip | `config.go` train options |
| Mini-batch training with shuffle | `config.go` train options |
| Training loop with forward + backward + Adam step | Transformer.Fit, Network.Fit |
| XOR convergence under SGD+momentum | `TestFit_SGDMomentum_ConvergesOnXOR` |
| Loss-decreases smoke test for transformer | `TestTransformer_FitLossDecreases` |

### Persistence
| Area | Notes |
|------|-------|
| v0 .relux format (gob) | `internal/serialize/gob.go` — legacy Network; bf16 in v0 LayerData too |
| v1 .relux format | `internal/serialize/v1.go`, `v1_read.go` — "RELV" magic, CRC32 header, SHA-256 footer, bf16 weights, f32 optimizer state |
| Magic-byte sniff | Network.Load rejects v1; LoadTransformer rejects v0 |
| Optimizer state round-trip | Adam m/v restored losslessly across save / load |
| End-to-end training checkpoint | `TestTransformer_V1EndToEndTraining` — train, save, load, resume |

### Backend abstraction
| Area | Notes |
|------|-------|
| `internal/compute/interface.go` | `ComputeBackend` — `MatMulFloat32` for the active bf16 path, legacy `MatMul` for compat |
| Native (pure Go) backend | `internal/compute/native.go` — always available |
| rnxa backend (CGO-free shims) | `internal/compute/rnxa.go` + `rnxa_stub.go` — `libmps.dylib` / `libcuda.so` via purego |
| Enhanced rnxa backend | `internal/compute/enhanced_rnxa.go` — pooled tensors + batch jobs |
| Auto-dispatch ladder | MPS → Metal → CPU on macOS, CUDA → CPU on Linux, CPU on Windows |
| **bf16 matmul wired** | `matmulBatched3D` dispatches through `Backend.MatMulFloat32`; bf16→f32 widen once per call; `NewTransformer` auto-creates the backend |

### Other
| Area | Notes |
|------|-------|
| Off-heap tensor storage | `internal/alloc/` backed by `xDarkicex/memory` |
| Network introspection | `Summary`, `Architecture`, `Validate`, `ParameterCount` |
| Concurrency-safe rng | per-layer LCG for deterministic init; no global rand state |

---

## In Flight 🔄

| Area | Target | Notes |
|------|:------:|-------|
| BPE tokenizer | 2026 Q3 | Required for real text training; the current toy examples use integer token IDs directly. A small `internal/tokenizer/bpe.go` plus a `relux.Tokenizer` type and `LoadVocab` / `Encode` / `Decode` API. |
| CLI (`cmd/relux`) | 2026 Q3 | `relux train`, `relux generate`, `relux inspect` — drives the public `Transformer` API. Reads v1 .relux files. |
| Multi-Token Prediction (MTP) loss | 2026 Q3 | Current Transformer.TrainStep does single-token next-token prediction; MTP (predict tokens k+1, k+2, … simultaneously) is the standard richer supervision signal. Drop-in for the loss path. |
| KV-cache | 2026 Q3 | `internal/transformer/kvcache.go` exists but isn't wired into `Transformer.Generate`; the public API still does the slow per-token prefill. |
| Per-axis Sum / Mean on GPU | 2026 Q4 | rnxa-side; the layernorm / softmax reduction loops are pure Go today. |
| CUDA end-to-end validation | 2026 Q4 | Build agent with `nvcc` + an NVIDIA GPU; runs `internal/compute/cuda/cuda_test.go` and the full rnxa test sweep. |

---

## Planned 🔮

| Area | Target | Notes |
|------|:------:|-------|
| Quantization (int8 / int4) for inference | 2026 Q4 | PTQ on bf16 weights; the v1 format can carry quantized weights alongside the bf16 master. |
| Windows CUDA / DirectML via rnxa | 2027 Q1 | `cuda_engine_windows.go` + `device_windows.go`; the C ABI is portable from the Linux cut |
| Convolutional (Conv1D / Conv2D) layers | 2027 Q1 | Im2col + rnxa MatMul under the hood; needed for vision-language extensions |
| Pooling (max, average) layers | 2027 Q1 | Companion to Conv layers |
| Grouped Query Attention variants | 2027 Q1 | Already supported (numKVHeads < numHeads), but the public API only exposes the standard config — extend the docs and add a config helper |
| Speculative decoding | 2027 Q2 | A draft model proposes k tokens, the target verifies in one forward — cuts generation latency 2-3×. |
| Mixture-of-Experts (MoE) | 2027 Q2 | Sparse FFN block; top-k routing; standard for modern frontier models. |
| Distributed training (data-parallel) | 2027 Q3 | Multi-process with a parameter server; gRPC transport is the candidate. |
| ONNX import | 2027 Q4 | Read a model, lower into the relux layer graph |
| ONNX export | 2027 Q4 | Companion to import |
| Multi-GPU rnxa backend | 2027 Q2 | Single-host, multiple MPS / CUDA devices |
| Flash Attention 2 | 2027 Q2 | Tile-based attention with O(N) memory; pure Metal shader on the rnxa side |
| Enterprise SLA / commercial support | 2027+ | No public timeline yet |

---

## Recent Milestones (reverse chronological)

- **2026 Q2** — **rnxa compute backend wired for bf16.** Added
  `MatMulFloat32` to `ComputeBackend`; `matmulBatched3D` dispatches
  through it, widening bf16→f32 once per call. `NewTransformer`
  auto-creates the backend. On macOS with `-tags rnxa`, the MPS engine
  (Apple AMX) runs the matmul at native hardware speed; the Metal
  compute shader is the fallback. Pure Go is always available.
- **2026 Q2** — **bf16 mixed precision refactor.** The original v1 format treated
  bf16 as wire-format compression; the active weight in memory was float64. The
  refactor changed `optim.Param.Data` from `[]float64` to `[]uint16` (bf16)
  and `optim.Param.Grad` to `[]float32`. Adam's m/v moved from float64 to
  float32 (bf16 would quantize the running averages and cause loss spikes
  on resume). The transformer layers and the MLP/Dense layer were both
  migrated. Result: the matmul now runs on bf16 weights with f32
  accumulation — the same pattern used by NVIDIA Tensor Cores and Apple AMX.
  Wire format unchanged for weights (bf16); Adam state on disk is now
  float32 instead of bf16.
- **2026 Q2** — **.relux v1 binary format.** 32-byte header (magic "RELV",
  version, num_layers, total_params, CRC32) + body (arch block, bf16
  weight block, float32 Adam state block) + 32-byte SHA-256 footer.
  Corruption detection: header CRC32 catches truncation; body SHA-256
  catches bit-flips. Magic-byte sniff in `Network.Load` dispatches
  v0 (gob) vs v1.
- **2026 Q1** — `relux.Transformer` first cut. RoPE, MHA with GQA, RMSNorm,
  MLP, Block, Embedding, Linear. Forward + backward + Adam. Sampler for
  generation. The `relux.Transformer` type lives in `transformer.go` and
  composes the `internal/transformer/` modules.
- **2026 Q1** — `optim` package split out; Adam shipped behind
  `relux.Optimizer(optim.Adam{LR: 0.001})`; SGD with momentum still works
  through `relux.Momentum(0.9)`.
- **2025 Q4** — OneDNN CPU shim built and unit-tested on darwin; the
  dispatcher's `try-and-continue` ladder now falls through cleanly when a
  backend is unavailable.
- **2025 Q3** — `internal/compute/` refactored: `interface.go`, `factory.go`,
  `pool.go`; `enhanced_rnxa.go` and `rnxa_stub.go` provide the
  prefer-rnxa-with-fallback policy.
- **2025 Q2** — Model persistence and optimizer state round-trip through gob.

---

## Architectural Decisions (load-bearing)

These are decisions that affect every line of code; recorded here so
future contributors don't accidentally undo them.

1. **bf16 is the active execution format, not just storage.** Weights
   are bf16 in memory, widened to f32 per matmul multiply, accumulated
   in f32. This is the industry-standard mixed-precision pattern
   (LLaMA, Mistral, MiniMax). Don't propose float16, fp8, or any
   non-bf16 hybrid without strong justification.
2. **Adam state is float32, not bf16.** The running averages `m = β₁m
   + (1-β₁)g` and `v = β₂v + (1-β₂)g²` rely on accumulating very
   tiny precise fractions (with β₂=0.999, the `(1-β₂)·g²` term is
   ~0.001·g²). bf16 truncation of these would cause loss spikes on
   checkpoint resume.
3. **The MLP and Transformer share `optim.Param`.** The bf16/f32
   contract applies to both. There is no float64 path.
4. **v1 is the canonical wire format for new code.** v0 (gob) is
   read-only back compat for old `Network` saves. New work writes
   v1.
5. **The `ComputeBackend` interface now has `MatMulFloat32`.** The
   active compute path is float32. Weights are bf16 in memory,
   widened to f32 once per matmul call (not per multiply), then
   dispatched through the backend. The rnxa MPS engine gets float32
   tensors natively — no float64 conversion. `NewTransformer` auto-
   creates the backend; set `transformer.Backend = nil` to force
   pure Go (e.g. for deterministic testing).

---

## How to use this document

- If you're picking up an in-flight item, the milestone notes link
  to the relevant code path.
- If you're scoping a new feature, the architectural decisions
  section is the place to start.
- If you're writing a release note, the recent milestones section
  is the source of truth (rather than the README, which can lag).
