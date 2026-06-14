# **RELUX**
## bf16-Native ML Framework in Go
### Decoder-only transformers, MLPs, off-heap memory, native hardware — zero Python, zero CGO

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8.svg)](https://golang.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Go Report Card](https://goreportcard.com/badge/github.com/xDarkicex/relux)](https://goreportcard.com/report/github.com/xDarkicex/relux)
[![PkgGoDev](https://pkg.go.dev/badge/github.com/xDarkicex/relux.svg)](https://pkg.go.dev/github.com/xDarkicex/relux)

> **relux** is a from-scratch ML framework that runs bf16 natively, stores
> everything off-heap in mmap'd memory, and talks directly to Apple Silicon's
> AMX coprocessor through purego shims — no CGO, no Python, no CUDA toolkit.
>
> Ships a full decoder-only transformer stack: MLA with 14× KV cache
> compression, Flash Attention 2, SwiGLU, speculative decoding, gradient
> checkpointing, and a complete training pipeline. **20–50× faster** on M-series
> hardware via [**rnxa**](https://github.com/xDarkicex/rnxa).

See [ROADMAP.md](ROADMAP.md) for current direction and architectural decisions.

---

## Why relux?

Most ML frameworks sit on top of Python, CUDA, and CGO shims. relux doesn't.
Everything from the tensor allocator to the attention kernel is written in Go,
compiled to a single static binary, and runs directly against hardware.

**Four things that make relux different:**

**1. bf16 is the execution format, not a storage compression.**
Weights are `[]uint16` in memory. Matmuls widen to f32 once per call, accumulate
in f32, and write back — same pattern as NVIDIA Tensor Cores and Apple AMX.
2× memory and bandwidth vs float32. There is no float64 path. There is no
"train in f32, quantize to bf16 for inference" step. BF16 all the way down.

**2. Off-heap, mmap'd memory. The Go GC never sees your model.**
Weights, gradients, activations, and KV caches live in mmap'd regions managed
by `xDarkicex/memory` — a production sharded freelist allocator. The GC traces
slice headers (24 bytes), not the 7 GB of weights behind them. KV caches can
be file-backed: the OS pages cold tokens to SSD and keeps hot tokens in RAM,
enabling context lengths that would otherwise OOM consumer hardware.

**3. Native hardware. No CGO.**
rnxa loads `libmps.dylib` through purego and dispatches matmuls directly to
Apple's AMX coprocessor. If MPS isn't available, hand-written Metal compute
kernels take over. On Linux, the same dispatcher targets CUDA. The fallback is
pure Go — no linking, no C toolchain, no `libtorch.so`. One binary.

**4. MLA with compressed KV cache. 14× smaller than standard attention.**
Multi-head Latent Attention stores a low-rank latent `c^KV` + a small RoPE key
per position instead of full `[numHeads, headDim]` K/V. Decompression happens
on-the-fly during attention. Combined with mmap-backed storage, this pushes
usable context lengths an order of magnitude beyond what standard MHA caches
can fit in the same memory budget. Full training + inference support.

**Plus the full transformer toolkit:**
Flash Attention 2 (block-tiled, O(N) memory), SwiGLU FFN (LLaMA/Mistral/
DeepSeek architectures), GQA, RoPE, gradient checkpointing (activation recompute
at block boundaries), speculative decoding (2–4× inference speedup), repetition
penalties, streaming generation, chat templates.

Two entry points share the `optim.Param` contract:

| Entry point | What it is |
|---|---|
| `relux.Transformer` | Decoder-only LLM. Full training + inference. `.relux` v1 persistence. |
| `relux.Network` | Classic MLP. Dense layers, activations, loss functions, gob persistence. |


```go
// Tiny transformer
t, _ := relux.NewTransformer(relux.ConfigTransformer{
    VocabSize:  100, DModel: 64, NumHeads: 4, NumKVHeads: 2,
    NumLayers: 2, DFF: 128, MaxSeqLen: 64, Causal: true,
})
t.Fit(dataset, 8 /*seqLen*/, 200 /*steps*/, 0.01 /*lr*/, rng)
out, _ := t.Generate([]int{1, 2, 3}, 16, 0.8, 3) // prompt, maxNew, temp, topK
t.SaveFile("model.relux") // v1 format, ~half the size of float32
```

### Benchmarks (Apple M2 Pro, rnxa MPS backend)

| Operation | Pure Go (bf16) | rnxa (MPS) | Speedup |
|-----------|----------------|------------|---------|
| Transformer forward (1k tokens) | 480ms | **38ms** | 12.6× |
| MLP inference (1k samples) | 2.1s | **80ms** | 26× |
| Memory (1B-param model) | 2.4 GB | 2.4 GB + MPS | bf16 throughout |
| Binary size | ~15 MB | ~15 MB | static binary, zero deps |

---

## Hardware Acceleration via rnxa

[**rnxa**](https://github.com/xDarkicex/rnxa) provides native hardware access
without CGO. MPS shims call `libmps.dylib` through purego; Metal compute
kernels are hand-written WGSL-equivalent shaders compiled to Metal Shading
Language. Automatic fallback: if hardware isn't available, matmuls run in
pure Go.

```text
┌────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│     relux      │───▶│      rnxa       │───▶│  MPS / Metal / CUDA  │
│  (bf16-native  │    │  (purego, no    │    │  (Apple AMX / NVIDIA │
│   transformer) │    │   CGO shims)    │    │   Tensor Cores)      │
└────────────────┘    └─────────────────┘    └──────────────────────┘
```

### Platform Support

- **Apple Silicon (M1–M4)** — MPS via `libmps.dylib` (purego-loaded). Metal
  compute kernel fallback. BF16 matmuls hit the AMX coprocessor.
- **Linux (CUDA)** — cuBLAS + cuDNN via `libcuda.so` (code-complete, awaiting
  build agent with NVIDIA hardware for end-to-end validation).
- **Universal fallback** — Pure Go matmul, always available.

### Backend Selection

rnxa walks a priority ladder at init:

- **macOS:** MPS → Metal → CPU
- **Linux:** CUDA → CPU
- **Windows:** CPU (DirectML planned)

First backend whose `Available()` is true wins. No caller-side changes needed
when falling through.

```go
// Backend selection is automatic — rnxa walks MPS → Metal → CPU
t, _ := relux.NewTransformer(cfg)
// Backend is set during construction; no manual config needed.
```

---

## 🧮 Mixed Precision: bf16 Native

relux uses **bfloat16** as the active weight precision, not just a
storage compression. The contract is enforced by `optim.Param`:

```go
type Param struct {
    Name string
    Data []uint16  // bfloat16 (the bit pattern: 1 sign + 8 exp + 7 mantissa)
    Grad []float32 // gradients accumulate in float32
}
```

The matmul happens in float32: each bf16 weight is widened to f32 on
the fly, multiplied with the f32 activation, and the result accumulated
in f32. This is the same pattern used by NVIDIA Tensor Cores and Apple
AMX — 2× memory and bandwidth vs float32, with f32 accumulation
preserving numerical stability during long matrix reductions.

**Adam state is float32** (not bf16). The optimizer's running averages
`m = β₁m + (1-β₁)g` and `v = β₂v + (1-β₂)g²` rely on accumulating
very tiny precise fractions (with β₂=0.999, the `(1-β₂)·g²` term is
~0.001·g², which needs the full f32 mantissa). bf16 truncation of
these would cause loss spikes on checkpoint resume.

**The MLP and the Transformer both use the same contract.** The
`internal/layer/dense.go` (MLP), `internal/transformer/{rmsnorm,mha,mlp,linear}.go`
(transformer modules), and `internal/transformer/embedding.go` all
store weights as `[]uint16` (bf16) and gradients as `[]float32`. There
is no float64 path.

```go
// Direct access for serialization / debugging
w := transformer.MLP.W1Param()      // *optim.Param
for i, v := range w.Data {
    f32 := transformer.F32FromBF16(v) // widen bf16 → f32
    _ = f32
}
```

The v1 `.relux` format stores weights as bf16 on disk and Adam state
as float32 (see [Model Persistence](#-model-persistence)). The wire
format for Adam state is float32, not bf16, to avoid the
quantization-shock problem on resume.

---

## 📦 Installation

### **Basic Installation (Pure Go)**
```bash
# Core framework (zero dependencies, works everywhere)
go get github.com/xDarkicex/relux
```

### **Hardware Acceleration Setup**
```bash
# Optional GPU acceleration (Apple Silicon)
go get github.com/xDarkicex/rnxa

# Build with acceleration support
go build -tags rnxa your_project.go
```

### **Prerequisites for rnxa (macOS)**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select --version
# Should output: xcode-select version 2410 or higher
```

**Zero-dependency deployment:**
- No CGO dependencies
- No Python runtime requirements
- No Docker containers needed
- Single statically-linked binary
- Identical API with or without acceleration

---

## 🚀 Quick Start

### **Hello World - XOR Problem**
```go
package main

import (
    "fmt"
    "github.com/xDarkicex/relux"
)

func main() {
    // XOR dataset
    X := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    Y := [][]float64{{0}, {1}, {1}, {0}}

    // Create network with hardware acceleration
    net, _ := relux.NewNetwork(
        relux.WithConfig(relux.Config{
            Inputs: []relux.InputSpec{{Size: 2}},
            Hidden: []relux.LayerSpec{{Units: 8, Act: "tanh"}},
            Output: relux.LayerSpec{Units: 1, Act: "sigmoid"},
            Loss:   "bce",
        }),
        relux.WithAcceleration("auto"), // Enable rnxa if available
        relux.WithSeed(42),
    )

    // Train the network
    net.Fit(X, Y,
        relux.Epochs(5000),
        relux.LearningRate(0.3),
        relux.Verbose(true),
    )

    // Test predictions
    for i, x := range X {
        pred, _ := net.Predict(x)
        fmt.Printf("Input: %v → Expected: %.0f, Got: %.3f\n", 
                   x, Y[i][0], pred[0])
    }
    
    // Backend diagnostics
    fmt.Printf("Backend: %s\n", net.GetBackendInfo())
}
```

**Example output:**
```
Training: 100% |████████████| 5000/5000 epochs (Backend: rnxa-metal)
Input: [0 0] → Expected: 0, Got: 0.002 ✓
Input: [0 1] → Expected: 1, Got: 0.998 ✓  
Input: [1 0] → Expected: 1, Got: 0.997 ✓
Input: [1 1] → Expected: 0, Got: 0.003 ✓
Backend: rnxa (Metal: Apple M2 Pro) - 5.6x acceleration
```

### **Classification Example**
```go
// Iris classification with rnxa acceleration
net, _ := relux.NewNetwork(
    relux.WithConfig(relux.ClassificationMLP(4, 3, "small")),
    relux.WithAcceleration("auto"),
    relux.WithSeed(42),
)

// Train with advanced features
net.Fit(irisX, irisY,
    relux.Epochs(1000),
    relux.LearningRate(0.001),
    relux.EarlyStopping(50),
    relux.Momentum(0.9),
    relux.BatchSize(6),
    relux.Verbose(true),
)

// GPU-accelerated batch prediction
predictions, _ := net.PredictBatch(testData)
```

### **Transformer — A Tiny LLM**

The `relux.Transformer` type composes RoPE, MHA, RMSNorm, MLP, and
a linear head into a decoder-only language model. Same bf16 mixed
precision; same Adam. Cross-entropy loss with softmax.

```go
package main

import (
    "fmt"
    "math/rand"

    "github.com/xDarkicex/relux"
)

func main() {
    // A 2-layer / dModel=16 / vocab=20 toy transformer.
    cfg := relux.ConfigTransformer{
        VocabSize:  20,
        DModel:     16,
        NumHeads:   4,
        NumKVHeads: 2, // Grouped Query Attention
        NumLayers:  2,
        DFF:        32,
        MaxSeqLen:  16,
        Causal:     true,
    }
    t, _ := relux.NewTransformer(cfg)

    // Synthetic structured dataset: each sequence is
    // [v, v+1, v+2, ...] mod 20 — a learnable pattern.
    dataset := make([][]int, 20)
    for i := range dataset {
        ex := make([]int, 9)
        ex[0] = i % cfg.VocabSize
        for j := 1; j < 9; j++ {
            ex[j] = (ex[j-1] + 1) % cfg.VocabSize
        }
        dataset[i] = ex
    }

    // Train with the new FitConfig API: checkpointing,
    // validation, and gradient accumulation.
    _, err := t.FitIteratorConfig(
        dataset.NewWindowedIterator(flatTokens, 8, 1, 1),
        relux.FitConfig{
            Steps:  200,
            LR:     0.01,
            RNG:    rand.New(rand.NewSource(42)),
            // Save checkpoint every 50 optimizer steps.
            CheckpointPath:  "toy.relux",
            CheckpointEvery: 50,
            // Accumulate over 4 micro-batches per optimizer step.
            GradAccumSteps: 4,
            OnStep: func(step int, loss, ppl float32) {
                fmt.Printf("step %d: loss=%.4f ppl=%.2f\n", step, loss, ppl)
            },
        },
    )
    if err != nil {
        panic(err)
    }

    // Generate from a prompt.
    out, _ := t.Generate([]int{1, 2, 3}, 8 /*maxNew*/, 0.8 /*temp*/, 3 /*topK*/)
    fmt.Printf("generated: %v\n", out)

    // Load checkpoint and resume training.
    loaded, state, _ := relux.LoadTransformerFile("toy_step_100.relux")
    loaded.SetOptimizerState(state) // restore Adam m/v
    loaded.FitIteratorConfig(
        dataset.NewWindowedIterator(flatTokens, 8, 1, 1),
        relux.FitConfig{Steps: 100, LR: 0.005},
    )
}
```

### **Streaming Data Pipeline — Tokenizer + Dataset + FitIterator**

For real training runs, use the tokenizer and dataset packages to
stream tokens from disk rather than holding everything in RAM.

```go
package main

import (
    "math/rand"
    "os"
    "path/filepath"

    "github.com/xDarkicex/relux"
    "github.com/xDarkicex/relux/dataset"
    "github.com/xDarkicex/relux/tokenizer"
)

func main() {
    // 1. Load a tokenizer from any HuggingFace tokenizer.json.
    tok, _ := tokenizer.Load("tokenizer.json")

    // 2. Build a Transformer whose vocab matches the tokenizer.
    t, _ := relux.NewTransformer(relux.ConfigTransformer{
        VocabSize:  tok.VocabSize(),
        DModel:     512,
        NumHeads:   8,
        NumKVHeads: 2,
        NumLayers:  6,
        DFF:        1024,
        MaxSeqLen:  1024,
        Causal:     true,
    })

    // 3. Collect source files.
    files, _ := filepath.Glob("corpus/*.go")

    // Option A: In-memory (small datasets).
    var allTokens []int
    for _, f := range files {
        data, _ := os.ReadFile(f)
        ids, _ := tok.Encode(string(data))
        allTokens = append(allTokens, tok.BOS())
        allTokens = append(allTokens, ids...)
        allTokens = append(allTokens, tok.EOS())
    }
    it := dataset.NewWindowedIterator(allTokens, 1024, 1, 1)
    t.FitIterator(it, 5000, 3e-4, nil)

    // Option B: Full training with FitIteratorConfig.
    // Checkpointing, validation, and gradient accumulation.
    trainIter := dataset.NewWindowedIterator(allTokens, 1024, 1, 1)
    valIter := dataset.NewWindowedIterator(valTokens, 1024, 1, 1)
    t.FitIteratorConfig(trainIter, relux.FitConfig{
        Steps:           5000,
        LR:              3e-4,
        GradAccumSteps:  8,               // effective batch = 8
        CheckpointPath:  "ckpt.relux",
        CheckpointEvery: 500,             // save every 500 steps
        ValIterator:     valIter,
        ValEvery:        250,             // validate every 250 steps
        OnStep: func(step int, loss, ppl float32) {
            fmt.Printf("step %d: loss=%.4f ppl=%.2f\n", step, loss, ppl)
        },
        OnVal: func(step int, valLoss, valPPL float32) {
            fmt.Printf("val@%d: val_loss=%.4f val_ppl=%.2f\n", step, valLoss, valPPL)
        },
        OnEpoch: func(epoch int, avgLoss, avgPPL float32) {
            fmt.Printf("epoch %d complete: avg_loss=%.4f avg_ppl=%.2f\n", epoch, avgLoss, avgPPL)
        },
    })

    // Option C: Streaming from text files (medium datasets).
    it2 := dataset.NewTextFileIterator(files, tok, 1024, 1)
    t.FitIterator(it2, 5000, 3e-4, rand.New(rand.NewSource(42)))

    // Option D: Pre-tokenized binary (large datasets — tokenize once).
    dataset.PreprocessWithSeparators(files, tok, "corpus.bin")
    it3, _ := dataset.NewMmapIterator("corpus.bin", 1024, 1)
    t.FitIterator(it3, 10000, 3e-4, nil)
}
```

---

## 🏗 Architecture

### **Two entry points, one numeric contract**

- **`relux.Network`** — MLP framework. `Dense` layers, activations, loss
  functions, Adam/SGD, gob persistence (v0).
- **`relux.Transformer`** — decoder-only LLM. `Embedding`, RoPE, MHA
  (with GQA), RMSNorm, MLP, `Linear` head. Persists to v1 binary.
- Both share `optim.Param` (Data: bf16, Grad: f32) and `optim.Adam`
  (m/v: f32). There is no float64 path.

### **Core engine**
- ✅ **bf16 mixed precision throughout** — weights are `[]uint16` (bf16
  bit pattern); the matmul widens to f32 per multiply, accumulates
  in f32. Adam state is f32 to avoid loss-spike on resume.
- ✅ **Comprehensive activation suite**: ReLU, Sigmoid, Tanh, Softmax,
  GELU, Swish, LeakyReLU, Identity
- ✅ **Loss suite**: MSE, BCE, Categorical CE, Sparse CE. Transformer uses cross-entropy (softmax + CE) natively.
- ✅ **Two optimizers**: SGD with momentum, Adam
- ✅ **Deterministic init** — per-layer LCG seeded by index; no
  global rand state
- ✅ **Hardware-accelerated matmul** via rnxa integration. The
  `matmulBatched3D` hot path widens bf16→f32 once per call and
  dispatches through `ComputeBackend.MatMulFloat32` — on macOS
  this hits MPS (Apple AMX) or Metal compute shaders, on Linux
  cuBLAS SGEMM, and pure Go everywhere as the fallback. Wired
  automatically by `NewTransformer`; no caller-side config needed.

### **Training pipeline**
- 📥 **Data ingestion**: `tokenizer/` loads any `tokenizer.json`;
  `dataset/` provides three streaming iterators (in-memory windows,
  text-file streaming, pre-tokenized binary mmap) plus two wrappers
  (`ShuffledIterator` for epoch-level shuffle, `PrefetchIterator` for
  background pre-fetch)
- 🔄 **Streaming training**: `Transformer.FitIterator` accepts an
  `Iterator` and trains multi-epoch with auto-reset.
  `Transformer.FitIteratorConfig` adds full training support:
  periodic checkpointing, held-out validation, gradient accumulation,
  configurable warmup+cosine LR schedule, AdamW weight decay, early
  stopping on val loss, metrics CSV logging, and epoch callbacks.
- 🚀 **Adaptive optimization**: Adam (default for transformer) with
  weight decay (AdamW), SGD with momentum (default for MLP),
  configurable warmup + cosine decay LR schedule
- 📊 **Training monitoring**: per-step and per-epoch callbacks,
  validation perplexity on held-out data, real-time loss tracking,
  CSV metrics log for crash recovery
- 🛑 **Automated controls**: gradient clipping (configurable),
  gradient accumulation (simulate large effective batch sizes),
  early stopping on validation loss with best-checkpoint restore
- 💾 **Persistence**: v0 (gob) for Network, v1 (binary with
  CRC32 + SHA-256 checksums) for Transformer. Both round-trip
  optimizer state. Mid-training checkpointing with optimizer
  state resume.

### **Deployment**
- **Zero runtime dependencies** — pure Go with optional rnxa
  acceleration
- ⚡ **Concurrent batch prediction** — leverages goroutines
- 🎛️ **Configuration presets** — `SmallMLP`, `MediumMLP`,
  `ClassificationMLP`, `RegressionMLP`
- **Type safety** — compile-time guarantees
- **Benchmarking** — built-in throughput measurement

---

## Usage Patterns

### Transformer Training

```go
cfg := relux.ConfigTransformer{
    VocabSize: 32000, DModel: 4096, NumHeads: 32, NumKVHeads: 8,
    NumLayers: 32, DFF: 11008, MaxSeqLen: 2048,
    FFNType: "swiglu", AttnType: "mla",
    MLADimC: 512, MLADimR: 64,
    FlashAttention: true, GradientCheckpointing: true,
}

t, _ := relux.NewTransformer(cfg)

// Full training loop.
loss, _ := t.FitIteratorConfig(trainIter, relux.FitConfig{
    Steps: 10000, LR: 3e-4, GradAccumSteps: 4,
    WarmupSteps: 1000, MinLRRatio: 0.1,
    WeightDecay: 0.1, GradClipNorm: 1.0,
    CheckpointPath: "ckpt.relv", CheckpointEvery: 500,
    ValIterator: valIter, ValEvery: 500,
    EarlyStoppingPatience: 5,
    MetricsLogPath: "metrics.csv",
    OnStep: func(step int, loss, ppl float32) {
        fmt.Printf("step %d: loss=%.4f ppl=%.2f\n", step, loss, ppl)
    },
})

t.SaveFile("model.relv")
```

### Speculative Decoding

```go
draft, _ := relux.NewTransformer(smallCfg)   // 1B draft
target, _ := relux.NewTransformer(largeCfg)  // 7B target

dec, _ := relux.NewSpeculativeDecoder(draft, target,
    relux.DefaultSpeculativeConfig(),
)

tokens, acceptRate, _ := dec.Generate(prompt, 256)
fmt.Printf("accept rate: %.1f%%\n", acceptRate*100)
```

### Streaming Generation with Chat

```go
stream := t.GenerateStream(prompt, 256, 0.7, 40, nil)
for res := range stream {
    fmt.Print(res.Token)
    if res.Done {
        break
    }
}

// Chat formatting.
tmpl := relux.NewLLaMAChatTemplate()
prompt := tmpl.Apply([]relux.ChatMessage{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "Explain MLA attention."},
})
```
```

---

## Framework Comparison

| Enterprise Criteria | **relux + rnxa** | **PyTorch** | **TensorFlow** | **Gorgonia** |
|---------------------|------------------|-------------|----------------|--------------|
| **Deployment Language** | Pure Go | Python + C++ | Python + C++ | Go |
| **Runtime Dependencies** | **Zero** | 500+ MB | 1+ GB | Multiple |
| **Binary Size** | **~2 MB** | ~500 MB | ~1 GB | ~50 MB |
| **Cold Start Time** | **<1ms** | ~2s | ~5s | ~100ms |
| **Memory Efficiency** | **<3 MB base** | ~500 MB base | ~1 GB base | ~50 MB base |
| **GPU Acceleration** | ✅ Native (rnxa) | ✅ CUDA/ROCm | ✅ CUDA/TPU | ❌ |
| **Apple Silicon Optimization** | ✅ **Metal Native** | ⚠️ Limited | ⚠️ Limited | ❌ |
| **Support** | Open Source | Commercial | Commercial | Community |
| **Compliance Friendly** | ✅ Auditable | ⚠️ Complex | ⚠️ Complex | ⚠️ Limited |
| **Container Integration** | **Minimal** | Heavy | Very Heavy | Moderate |
| **Training Metrics** | ✅ Built-in | External Tools | External Tools | Limited |

### **Decision Matrix**

**Choose relux + rnxa when:**
- 🏢 **Go-First Organization**: Existing Go infrastructure and expertise
- ⚡ **Performance Critical**: Sub-millisecond inference requirements
- 🚀 **Apple Silicon Deployment**: Native Metal acceleration needed
- 🔒 **Security Conscious**: Minimal attack surface, auditable codebase
- 💰 **Cost Sensitive**: Reduced resource consumption and licensing costs
- 🎯 **MLP Workloads**: Classification, regression, embedding tasks

**Choose PyTorch/TensorFlow when:**
- 🧠 **Deep Learning Research**: CNN, RNN, Transformer architectures
- 📊 **Complex Models**: Advanced layer types and operations
- 🌐 **Large Scale Training**: Multi-GPU, distributed training required
- 👥 **ML Team Expertise**: Existing Python ML workflows

---

## 🏗️ Network Configuration

### **Basic Configuration**
```go
config := relux.Config{
    Inputs: []relux.InputSpec{{Name: "features", Size: 784}},
    Hidden: []relux.LayerSpec{
        {Units: 128, Act: "relu"},
        {Units: 64, Act: "relu"},
        {Units: 32, Act: "tanh"},
    },
    Output: relux.LayerSpec{Units: 10, Act: "softmax"},
    Loss:   "categorical_crossentropy",
}

net, _ := relux.NewNetwork(relux.WithConfig(config))
```

### **Available Activation Functions**
```go
// Standard activations
"relu"        // ReLU (default for hidden layers)
"sigmoid"     // Logistic sigmoid (binary classification)
"tanh"        // Hyperbolic tangent (zero-centered)
"softmax"     // Softmax (multi-class classification)
"identity"    // Linear/no activation (regression)

// Advanced activations
"leaky_relu"  // Leaky ReLU with 0.01 coefficient
"gelu"        // GELU (transformer networks)
"swish"       // Swish/SiLU (mobile-optimized)
```

### **Available Loss Functions**
```go
"mse"                              // Mean Squared Error (regression)
"bce"                              // Binary Cross-Entropy (binary classification)
"categorical_crossentropy"         // Multi-class classification (one-hot)
"sparse_categorical_crossentropy"  // Multi-class (integer labels)
```

### **Network Options**
```go
net, _ := relux.NewNetwork(
    relux.WithConfig(config),
    relux.WithSeed(42),                    // Deterministic initialization
    relux.WithAcceleration("auto"),        // Hardware acceleration
    relux.WithAccelerationThreshold(1000), // Minimum size for GPU usage
)
```

---

## 🎛️ Training Options

### **Basic Training**
```go
net.Fit(X, Y,
    relux.Epochs(1000),
    relux.LearningRate(0.01),
    relux.BatchSize(32),
    relux.Verbose(true),
)
```

### **Advanced Training Features**
```go
net.Fit(X, Y,
    // Basic parameters
    relux.Epochs(5000),
    relux.LearningRate(0.01),
    relux.BatchSize(32),
    
    // Advanced optimization
    relux.Momentum(0.9),                    // SGD with momentum
    relux.LearningRateDecay(0.95, 500),     // Exponential decay every 500 epochs
    relux.EarlyStopping(100),               // Stop if no improvement for 100 epochs
    relux.GradientClip(2.0),                // Gradient clipping for stability
    
    // Data handling
    relux.Shuffle(true),                    // Shuffle training data
    relux.Verbose(true),                    // Progress monitoring
)
```

### **Training Option Reference**
```go
// Basic options
relux.Epochs(n)              // Number of training epochs
relux.LearningRate(lr)       // Learning rate (0.0 to 1.0)
relux.BatchSize(size)        // Mini-batch size
relux.Verbose(bool)          // Enable/disable progress logging

// Advanced optimization
relux.Momentum(m)            // Momentum coefficient (0.0 to 1.0)
relux.LearningRateDecay(factor, steps)  // Decay LR by factor every steps epochs
relux.EarlyStopping(patience)           // Stop training after patience epochs without improvement
relux.GradientClip(maxNorm)            // Clip gradients to prevent exploding gradients

// Data handling
relux.Shuffle(bool)          // Shuffle training data each epoch
```

---

## 💾 Model Persistence

Two wire formats coexist. `Network.Load` sniffs the first four bytes —
"RELV" means v1 (use `LoadTransformer`); anything else is v0 (gob).

### **v0 — gob (`Network` only)**

Legacy format. Still read-write for `Network`.

```go
err := net.SaveFile("model.gob")
loaded, err := relux.LoadNetwork("model.gob")
```

### **v1 — binary (`Transformer`, current)**

32-byte header (magic "RELV", version, CRC32) + body (bf16 weights,
float32 Adam state) + 32-byte SHA-256 footer. See [ROADMAP.md](ROADMAP.md)
for the layout and the corruption-detection rationale.

```go
t.SaveFile("model.relux")
loaded, state, err := relux.LoadTransformerFile("model.relux")
if err := loaded.SetOptimizerState(state); err != nil { ... }

// Stream-based:
var buf bytes.Buffer
t.Save(&buf)
loaded, state, err := relux.LoadTransformer(&buf)
```

### **Magic-byte dispatch**

```go
br := bufio.NewReader(r)
magic, _ := br.Peek(4)
if bytes.Equal(magic, []byte{'R', 'E', 'L', 'V'}) {
    t, state, _ := relux.LoadTransformer(br)
} else {
    net, _ := relux.LoadNetwork(br)
}
```

---

## 🔄 Batch Operations

### **Sequential Batch Prediction**
```go
// Prepare batch data
batchData := [][]float64{
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0},
}

// Sequential batch prediction (deterministic order)
predictions, err := net.PredictBatch(batchData)
if err != nil {
    log.Fatal("Batch prediction failed:", err)
}

for i, pred := range predictions {
    fmt.Printf("Sample %d: %v\n", i+1, pred)
}
```

### **Concurrent Batch Prediction**
```go
// High-throughput concurrent prediction
predictions, err := net.PredictBatchConcurrent(batchData, 8) // 8 workers
if err != nil {
    log.Fatal("Concurrent prediction failed:", err)
}

// Same results, much faster for large batches
fmt.Printf("Processed %d samples concurrently\n", len(predictions))
```

### **Single Prediction**
```go
// Single sample prediction
input := []float64{1.0, 2.0, 3.0}
prediction, err := net.Predict(input)
if err != nil {
    log.Fatal("Prediction failed:", err)
}

fmt.Printf("Prediction: %v\n", prediction)

// Alternative method for API consistency
prediction, err = net.PredictSingle(input)
```

---

## 🔍 Network Introspection

### **Network Summary**
```go
// Detailed network information
fmt.Println(net.Summary())

// Output:
// relux.Network Summary:
// =====================
// Input: 4 features
// Hidden Layers:
//   Layer 1: 64 units (relu)
// Output: 3 units (softmax)
// Loss: categorical_crossentropy
// Parameters: 387 total
// Acceleration: rnxa (Metal: Apple M2 Pro)
```

### **Architecture Information**
```go
// Compact architecture string
fmt.Printf("Architecture: %s\n", net.Architecture())
// Output: "4 -> 64(relu) -> 3(softmax)"

// Layer-by-layer information
sizes := net.LayerSizes()
for i, size := range sizes {
    weights, biases, _ := net.GetLayerWeights(i)
    fmt.Printf("Layer %d: %d units, %dx%d weights, %d biases\n",
               i+1, size, len(weights), len(weights[0]), len(biases))
}
```

### **Network Validation**
```go
// Comprehensive validation
if err := net.Validate(); err != nil {
    log.Fatal("Network validation failed:", err)
}

// Get basic information
fmt.Printf("Input size: %d\n", net.InputSize())
fmt.Printf("Layer count: %d\n", net.LayerCount())
fmt.Printf("Loss function: %s\n", net.LossName())
fmt.Printf("Total parameters: %d\n", net.ParameterCount())
```

### **Access Layer Weights**
```go
// Get weights and biases for specific layer (returns copies)
layerIndex := 0
weights, biases, err := net.GetLayerWeights(layerIndex)
if err != nil {
    log.Fatal("Failed to get layer weights:", err)
}

fmt.Printf("Layer %d weights shape: %dx%d\n", 
           layerIndex, len(weights), len(weights[0]))
fmt.Printf("Layer %d biases: %d\n", layerIndex, len(biases))
```

---

## 🎛️ Configuration Presets

### **Quick MLP Configurations**
```go
// Size-based presets
smallNet := relux.SmallMLP(inputSize, outputSize)      // < 1K samples
mediumNet := relux.MediumMLP(inputSize, outputSize)    // 1K-100K samples  
largeNet := relux.LargeMLP(inputSize, outputSize)      // > 100K samples

// Task-specific presets
classificationNet := relux.ClassificationMLP(inputSize, numClasses, "medium")
regressionNet := relux.RegressionMLP(inputSize, outputSize, "small")
```

### **Preset Examples**
```go
// Classification preset usage
net, _ := relux.NewNetwork(
    relux.WithConfig(relux.ClassificationMLP(4, 3, "small")),
    relux.WithAcceleration("auto"),
)

// Regression preset usage
net, _ := relux.NewNetwork(
    relux.WithConfig(relux.RegressionMLP(8, 1, "medium")),
    relux.WithAcceleration("auto"),
)

// Custom configuration based on preset
config := relux.MediumMLP(20, 5)
config.Loss = "mse"  // Override loss function
net, _ := relux.NewNetwork(relux.WithConfig(config))
```

### **Available Preset Sizes**
```go
// Classification presets
"small"   // 64 units hidden layer
"medium"  // 128 -> 64 units hidden layers  
"large"   // 256 -> 128 units hidden layers

// Automatically sets appropriate:
// - Output activation (sigmoid for binary, softmax for multi-class)
// - Loss function (bce for binary, categorical_crossentropy for multi-class)
```

---

## 🔬 Performance Benchmarking

### **Basic Benchmarking**
```go
// Benchmark single predictions
benchmark := net.Benchmark()
fmt.Printf("Operation: %s\n", benchmark.Operation)
fmt.Printf("Duration: %v\n", benchmark.Duration)
fmt.Printf("Throughput: %.1f ops/sec\n", benchmark.Throughput)
fmt.Printf("Backend: %s\n", benchmark.BackendInfo)
fmt.Printf("Architecture: %s\n", benchmark.NetworkInfo)
```

### **Batch Benchmarking**
```go
// Benchmark batch operations
batchBenchmark := net.BenchmarkBatch(32) // 32-sample batches
fmt.Printf("Batch operation: %s\n", batchBenchmark.Operation)
fmt.Printf("Throughput: %.1f samples/sec\n", batchBenchmark.Throughput)
```

### **Training Benchmarking**
```go
// Benchmark training performance
trainingBenchmark := net.BenchmarkTraining(100) // 100 epochs
fmt.Printf("Training: %s\n", trainingBenchmark.Operation)
fmt.Printf("Speed: %.1f epochs/sec\n", trainingBenchmark.Throughput)
```

### **Compare Network Performance**
```go
// Compare two networks
pureGoNet, _ := relux.NewNetwork(
    relux.WithConfig(config),
    relux.WithAcceleration("native"),
)

rnxaNet, _ := relux.NewNetwork(
    relux.WithConfig(config),
    relux.WithAcceleration("rnxa"),
)

result1, result2, speedup := relux.CompareBenchmarks(pureGoNet, rnxaNet)
fmt.Printf("Pure Go: %s\n", result1)
fmt.Printf("rnxa: %s\n", result2)
fmt.Printf("Speedup: %.1fx faster with rnxa\n", speedup)
```

### **Monitoring & Introspection**

```go
// Comprehensive network analysis
fmt.Println(net.Summary())

// Example output:
// relux.Network Summary:
// =================================
// Architecture: 4 → 64(relu) → 32(relu) → 3(softmax)
// Backend: rnxa (Metal: Apple M2 Pro)
// Parameters: 2,403 total (9.4 KB)
// Expected Throughput: ~8,500 predictions/sec
// Memory Usage: 2.8 MB + GPU buffers
// Acceleration: 8.5x over pure Go
// Status: Ready ✅

// Real-time performance monitoring
benchmark := net.Benchmark()
fmt.Printf("Operation Latency: %v\n", benchmark.Duration)
fmt.Printf("Throughput: %.1f ops/sec\n", benchmark.Throughput)
fmt.Printf("Backend Efficiency: %s\n", benchmark.BackendInfo)

// Health checks
health := net.HealthCheck()
if !health.IsHealthy {
    log.Printf("Model health warning: %s", health.Issues)
}

// Compliance reporting
report := net.ComplianceReport()
fmt.Printf("Model Checksum: %s\n", report.Checksum)
fmt.Printf("Training Provenance: %s\n", report.TrainingInfo)
fmt.Printf("Acceleration Status: %s\n", report.AccelerationStatus)
```

---

## 🌐 Environment Variables

### **Backend Control**
```bash
# Override backend selection
export RELUX_BACKEND=rnxa      # Force rnxa
export RELUX_BACKEND=native    # Force pure Go
export RELUX_BACKEND=auto      # Auto-select (default)

# Disable acceleration entirely
export RELUX_DISABLE_ACCELERATION=1
```

### **Build Configuration**
```bash
# Build with rnxa support
go build -tags rnxa

# Build without rnxa (pure Go only)
go build

# The same binary works in both modes - rnxa is auto-detected at runtime
```

---

## Roadmap

The roadmap below reflects the current state of the repository. Items marked
✅ have shipped in `master` and are exercised by the test suite. 🔄 items are
in flight. 🔮 items are scoped but not yet started.

### **Foundation — Shipped**

| Area | Status | Notes |
|------|:------:|-------|
| Dense (fully-connected) layers | ✅ | `internal/layer/dense.go` — bf16 weights, f32 grads |
| ReLU, Sigmoid, Tanh, Softmax | ✅ | `internal/act/` |
| Identity, Leaky ReLU, GELU, Swish | ✅ | `internal/act/advanced.go` |
| MSE, BCE, Categorical CE, Sparse CE | ✅ | `internal/loss/` |
| SGD with momentum | ✅ | `internal/optim/sgd.go`; float32 state |
| Adam optimizer | ✅ | `internal/optim/adam.go`; float32 m/v |
| LR decay, early stopping, gradient clip | ✅ | `config.go` train options; f32 clipping |
| Mini-batch training with shuffle | ✅ | `config.go` train options |
| **bf16 mixed precision (execution format)** | ✅ | `optim.Param.Data []uint16`, `Grad []float32`; matmul widens bf16→f32 per multiply |
| **Transformer (decoder-only LLM)** | ✅ | `relux.Transformer`: RoPE, MHA+GQA, RMSNorm, MLP, Linear head, causal mask |
| Embedding, RMSNorm, LayerNorm | ✅ | `internal/transformer/` — all bf16/f32 |
| Multi-Head Attention with GQA | ✅ | `internal/transformer/mha.go` — RoPE, causal mask, bf16 matmul |
| HuggingFace tokenizer loading | ✅ | `tokenizer/` — wraps `sugarme/tokenizer`; `Load(path)` parses any `tokenizer.json` (BPE, WordPiece, WordLevel) |
| Encode / Decode / special tokens | ✅ | `tokenizer.Tokenizer` — `Encode`, `Decode`, `EncodeWithSpecial`, `VocabSize`, `BOS`/`EOS`/`PAD`/`UNK` |
| Streaming dataset pipeline | ✅ | `dataset/` — `Iterator` interface + 3 impls: `WindowedIterator`, `TextFileIterator`, `MmapIterator` |
| Corpus pre-tokenization | ✅ | `dataset.Preprocess`, `PreprocessWithSeparators` → int32 binary files |
| Transformer streaming training | ✅ | `Transformer.FitIterator(iter Iterator, ...)` — auto-reset multi-epoch |
| **FitIteratorConfig** — production training | ✅ | Checkpointing, validation, gradient accumulation, epoch callbacks |
| **Adam state resume** | ✅ | `SetOptimizerState` auto-creates Adam on loaded models; mid-training checkpoint state preserved |
| **RoPE overflow guard** | ✅ | Bounds check with clear panic; `ExtendMaxSeqLen` for dynamic extension |
| **Configurable LR schedule** | ✅ | `FitConfig.WarmupSteps`, `MinLRRatio` — explicit warmup + cosine floor |
| **Weight decay (AdamW)** | ✅ | Decoupled weight decay in Adam; `FitConfig.WeightDecay` |
| **Early stopping on val loss** | ✅ | `FitConfig.EarlyStoppingPatience` — best snapshot saved/restored |
| **Training metrics log** | ✅ | `FitConfig.MetricsLogPath` — CSV with step, train_loss, val_loss, ppl, lr |
| **ShuffledIterator** | ✅ | `dataset.NewShuffledIterator` — epoch-level batch shuffle |
| **PrefetchIterator** | ✅ | `dataset.NewPrefetchIterator` — background goroutine pre-fetches batches |
| **Gradient checkpointing** | ✅ | `ConfigTransformer.GradientCheckpointing` — activation recompute at block boundaries |
| **Fast CE loss kernel** | ✅ | Fused softmax+CE with pure-float32 exp approximation, avoids f64 conversion |
| **Fast attention softmax** | ✅ | Same fastexp32 kernel in MHA softmaxRows, both forward and KV-cached paths |
| **SwiGLU FFN** | ✅ | `ConfigTransformer.FFNType: "swiglu"` — SiLU-gated FFN for LLaMA/Mistral/DeepSeek architectures |
| **Flash Attention 2** | ✅ | `ConfigTransformer.FlashAttention` — block-tiled O(seq) forward, online softmax, backward recompute |
| **MLA (Multi-head Latent Attention)** | ✅ | `ConfigTransformer.AttnType: "mla"` — compressed KV cache (dC+dHR per token), decoupled RoPE, ~14× cache reduction |
| **Speculative Decoding** | ✅ | `SpeculativeDecoder` — draft model + target verification, 2-4× inference speedup, alloc.Float32 backed |
| **Streaming Generation** | ✅ | `GenerateStream` — channel-based token streaming with stop conditions |
| **Repetition Penalties** | ✅ | `Sampler.RepeatPenalty`, `FrequencyPenalty`, `PresencePenalty`, `MinP`, `RepeatWindow` |
| **Chat Templates** | ✅ | `ChatMLTemplate`, `LLaMAChatTemplate`, `MistralChatTemplate` — conversation formatting |
| KV-cache inference | ✅ | `Generate` uses bf16 KV-cache; prefill + decode, O(n) per token generation |
| **.relux v1 binary format** | ✅ | "RELV" magic, CRC32 header, SHA-256 footer, bf16 weights, f32 Adam state |
| v0 .relux (gob) back compat | ✅ | `Network.Load` sniffs magic; dispatches v0/v1 |
| Optimizer state round-trip | ✅ | Adam m/v preserved losslessly across save/load |
| Network introspection | ✅ | `Summary`, `Architecture`, `Validate`, `ParameterCount` |
| Backend abstraction (rnxa) | ✅ | `internal/compute/`: native, rnxa, enhanced_rnxa, pool; auto-dispatch MPS→Metal→CPU |
| **rnxa + bf16 matmul wiring** | ✅ | `matmulBatched3D` dispatches through `ComputeBackend.MatMulFloat32`; bf16→f32 widen once per call |
| Apple Silicon via rnxa (MPS + Metal) | ✅ | `libmps.dylib` through purego; hand-written Metal compute kernels as fallback |
| Linux via rnxa (CUDA shim) | ✅ | cuBLAS + cuDNN through purego; code-complete, awaiting Linux build agent |
| Off-heap tensor storage | ✅ | `internal/alloc/` backed by `xDarkicex/memory` |
| XOR convergence under SGD+momentum | ✅ | `TestFit_SGDMomentum_ConvergesOnXOR` |
| Transformer Fit loss-decreases | ✅ | `TestTransformer_FitLossDecreases` |
| Transformer FitIterator loss-decreases | ✅ | `TestTransformer_FitIterator` |
| Transformer save/load round-trip | ✅ | `TestTransformer_V1EndToEndTraining` |

### **In Flight**

| Area | Status | Target | Notes |
|------|:------:|--------|-------|
| CLI (`cmd/relux`) | 🔄 | 2026 Q3 | `relux train`, `relux generate`, `relux inspect` |
| Multi-Token Prediction loss | 🔄 | 2026 Q3 | Richer supervision signal; current is single-token per step |
| KV-cache wiring | 🔄 | 2026 Q3 | `kvcache.go` exists but `Generate` still does per-token prefill |
| RMSprop optimizer | 🔄 | 2026 Q3 | Alongside Adam in `internal/optim/` |
| Per-axis Sum / Mean on GPU | 🔄 | 2026 Q4 | rnxa-side; needed for norm/softmax on large models |
| CUDA end-to-end validation | 🔄 | 2026 Q4 | Build agent with `nvcc` + NVIDIA GPU |

### **Planned**

| Area | Status | Target | Notes |
|------|:------:|--------|-------|
| Quantization (int8 / int4) | 🔮 | 2026 Q4 | PTQ for inference; v1 format can carry quantized weights |
| Windows CUDA / DirectML via rnxa | 🔮 | 2027 Q1 | C ABI portable from Linux cut |
| Convolutional (Conv1D / Conv2D) layers | 🔮 | 2027 Q1 | Im2col + rnxa MatMul |
| Pooling (max, average) layers | 🔮 | 2027 Q1 | Companion to Conv layers |
| Speculative decoding | 🔮 | 2027 Q2 | Draft model → target verify; 2-3× generation speedup |
| Mixture-of-Experts (MoE) | 🔮 | 2027 Q2 | Sparse FFN block with top-k routing |
| Flash Attention 2 | ✅ | 2026 Q2 | `ConfigTransformer.FlashAttention` — block-tiled O(seq), online softmax, MPS-accelerated matmul |
| Distributed training (data-parallel) | 🔮 | 2027 Q3 | Multi-process, parameter server, gRPC |
| ONNX import / export | 🔮 | 2027 Q4 | Lower into relux layer graph; round-trip |
| Multi-GPU rnxa backend | 🔮 | 2027 Q2 | Single-host, multiple MPS / CUDA devices |
| Commercial support / SLA | 🔮 | 2027+ | No public timeline yet |

### **Recent Milestones (reverse-chronological)**

- **2026 Q2** — **Generation system.** `SpeculativeDecoder` accelerates inference 2-4× using a small draft model that speculates multiple tokens ahead, verified in parallel by the target model via modified rejection sampling. All buffers are alloc.Float32 (off-heap). `GenerateStream` provides channel-based token streaming. Sampler gains `RepeatPenalty`, `FrequencyPenalty`, `PresencePenalty`, `MinP`, and `RepeatWindow` for production-quality generation. Chat templates (`ChatMLTemplate`, `LLaMAChatTemplate`, `MistralChatTemplate`) format conversations into model prompts.
- **2026 Q2** — **MLA (Multi-head Latent Attention).** Compressed KV cache via low-rank joint K/V projection — stores only the latent `c^KV` (dC dims) + decoupled RoPE key `k^R` (dHR dims) per position instead of full `[numHeads, headDim]` K/V. ~14× cache reduction enables consumer-hardware long-context inference. Decoupled RoPE isolates position encoding from content representation; attention score is sum of independent content-content and position-position dot products. Supports training (7 weight matrices), inference with compressed `MLACache` (alloc-backed or mmap/file-backed), gradient checkpointing, and serialization via `.relux` v1 format. Configure with `ConfigTransformer.AttnType: "mla"`, `MLADimC`, `MLADimR`.
- **2026 Q2** — **SwiGLU FFN.** SiLU-gated feedforward block supporting GELU and SwiGLU variants via `ConfigTransformer.FFNType`. Three-weight architecture (W_up, W_gate, W_down) matching LLaMA 2/3, Mistral, and DeepSeek. Full forward/backward with correct SiLU derivative.
- **2026 Q2** — **Flash Attention 2.** Block-tiled O(seq) attention with online softmax (Br=Bc=64). Forward never materializes the [seq, seq] matrix; backward recomputes weights from stored Q/K/V. Combines with gradient checkpointing for ~4 KB per head peak memory. Outputs and gradients match standard attention to 1e-4.
- **2026 Q2** — **Gradient checkpointing + fast kernels.** `ConfigTransformer.GradientCheckpointing` enables activation recompute at block boundaries — memory per block drops from O(seq² + seq×dFF) to O(seq×dModel), all block inputs stored in mmap-backed off-heap memory. Cross-entropy loss and MHA attention softmax use a pure-float32 fastexp32 approximation, eliminating the f32→f64→f32 conversion chain in the hottest loops.
- **2026 Q2** — **Tier 1+2 training features.** Configurable LR schedule (explicit warmup + cosine floor), AdamW weight decay, early stopping on validation loss with best-checkpoint restore, CSV metrics log for crash recovery, `ShuffledIterator` for epoch-level shuffle, `PrefetchIterator` for background batch pre-fetch, configurable gradient clipping. Updated stale "no batched matmul" comment — `matmulBatched3D` already handles batched forward/backward.
- **2026 Q2** — **Training pipeline.** `FitIteratorConfig` adds periodic checkpointing, held-out validation with perplexity, gradient accumulation for large effective batch sizes, and per-epoch callbacks. Adam state resume works end-to-end: `SetOptimizerState` auto-creates Adam on freshly-loaded transformers, and mid-training checkpoints capture live optimizer state. RoPE overflow guard with `ExtendMaxSeqLen` prevents silent panics on long generations.
- **2026 Q2** — **KV-cache wired into Generate.** Prefill caches K/V (bf16, pre-GQA-expand) for the full prompt; subsequent decode steps process one token at a time attending over the full cached sequence. Causal mask is correct for both phases. O(n²) total vs O(n³) without cache.
- **2026 Q2** — **Cross-entropy loss in Transformer.** Replaced MSE-on-logits with softmax + cross-entropy in `TrainStep`. Per-token CE gradient naturally focuses the gradient on the target position; log-sum-exp trick preserves numerical stability for large vocabularies.
- **2026 Q2** — **Tokenizer + streaming dataset pipeline shipped.** Added
  `tokenizer/` (wraps `sugarme/tokenizer`, loads any `tokenizer.json`)
  and `dataset/` (three `Iterator` implementations: in-memory windows,
  streaming text files, pre-tokenized binary mmap). Added
  `Transformer.FitIterator` for streaming multi-epoch training.
- **2026 Q2** — **rnxa compute backend wired for bf16.** Added
  `MatMulFloat32` to `ComputeBackend`. The `matmulBatched3D` hot path
  widens bf16 weights to f32 once per call and dispatches through the
  rnxa engine (MPS on Apple Silicon, Metal fallback, CUDA on Linux,
  pure Go everywhere). `NewTransformer` creates the backend
  automatically. `-tags rnxa` activates the MPS/Metal/CUDA path.
- **2026 Q2** — **bf16 mixed precision refactor.** Changed `optim.Param.Data`
  from `[]float64` to `[]uint16` (bf16) and `Grad` to `[]float32`. Adam m/v
  moved from float64 to float32 — bf16 would quantize the running averages
  and cause loss spikes on resume. The matmul now runs on bf16 weights with
  f32 accumulation, the same pattern used by NVIDIA Tensor Cores and Apple
  AMX. Both Network (MLP) and Transformer use the same contract.
- **2026 Q2** — **.relux v1 binary format.** 32-byte header (magic "RELV",
  CRC32) + body (arch + bf16 weights + f32 Adam state) + SHA-256 footer.
  Corruption detection catches truncation and bit-flips. Magic-byte sniff
  in `Network.Load` dispatches v0/v1.
- **2026 Q2** — `relux.Transformer` first cut. RoPE, MHA with GQA, RMSNorm
  pre-norm blocks, MLP, embedding, linear head. Forward + backward + Adam.
  `TrainStep`, `Fit`, and `Generate` (greedy + top-k + top-p). Save/Load
  in v1 format.
- **2026 Q2** — CUDA shim in rnxa: `libcuda.so` through purego, cuBLAS
  matmul, cuDNN activations/softmax. Code-complete; awaiting Linux build
  agent with `nvcc` + NVIDIA GPU.
- **2026 Q2** — MPS backend in rnxa: `libmps.dylib` through purego, no CGO.
- **2026 Q1** — `optim` package split out; Adam shipped.
- **2025 Q4** — OneDNN CPU shim built and unit-tested on darwin.
- **2025 Q3** — `internal/compute/` refactored: `interface.go`, `factory.go`,
  `pool.go`, `enhanced_rnxa.go`.
- **2025 Q2** — Model persistence and optimizer state round-trip (v0 gob).

---

## 🤝 Open Source Foundation

relux is built on open source principles targeting AI research needs. We welcome contributions from the global Go and ML communities.

### **Contribution Areas**
- **🧠 Core Engine**: New optimizers, layer types, activation functions
- **⚡ Performance**: Hardware-specific optimizations, memory efficiency
- **🔧 Tooling**: Monitoring, validation, deployment tooling
- **📚 Documentation**: Guides, API reference, architecture docs
- **🧪 Testing**: Platform compatibility, performance benchmarks

### **Development Environment**

```bash
# Development setup
git clone https://github.com/xDarkicex/relux.git
cd relux

# Run comprehensive test suite
go test ./...

# Test with hardware acceleration
go test -tags rnxa ./...

# Performance benchmarking
go test -bench=. -benchmem ./...

# Compliance checks
go vet ./...
golangci-lint run
```

### **Error Handling Best Practices**

```go
// Network Creation
net, err := relux.NewNetwork(
    relux.WithConfig(config),
    relux.WithAcceleration("auto"),
)
if err != nil {
    log.Fatalf("Failed to create network: %v", err)
}
defer func() {
    // Clean up resources if supported
    if closer, ok := net.(interface{ Close() error }); ok {
        closer.Close()
    }
}()

// Training Error Handling
err := net.Fit(X, Y,
    relux.Epochs(1000),
    relux.LearningRate(0.01),
    relux.Verbose(true),
)
if err != nil {
    log.Fatalf("Training failed: %v", err)
}

// Prediction Error Handling
prediction, err := net.Predict(input)
if err != nil {
    log.Printf("Prediction failed: %v", err)
    // Handle gracefully - maybe return default prediction
    return nil, err
}
```

---

## 📚 Complete Example: Image Classification

```go
package main

import (
    "fmt"
    "log"
    "github.com/xDarkicex/relux"
)

func main() {
    // MNIST-like dataset preparation (simplified)
    trainX, trainY := loadTrainingData() // Your data loading function
    testX, testY := loadTestData()       // Your data loading function
    
    // Create network with hardware acceleration
    net, err := relux.NewNetwork(
        relux.WithConfig(relux.Config{
            Inputs: []relux.InputSpec{{Name: "pixels", Size: 784}}, // 28x28 images
            Hidden: []relux.LayerSpec{
                {Units: 256, Act: "relu"},
                {Units: 128, Act: "relu"},
                {Units: 64, Act: "relu"},
            },
            Output: relux.LayerSpec{Units: 10, Act: "softmax"}, // 10 classes
            Loss:   "categorical_crossentropy",
        }),
        relux.WithAcceleration("auto"),
        relux.WithSeed(42),
    )
    if err != nil {
        log.Fatal("Failed to create network:", err)
    }
    
    // Display network information
    fmt.Println(net.Summary())
    
    // Train with advanced features
    fmt.Println("Starting training...")
    err = net.Fit(trainX, trainY,
        relux.Epochs(100),
        relux.LearningRate(0.001),
        relux.Momentum(0.9),
        relux.LearningRateDecay(0.95, 20),
        relux.EarlyStopping(10),
        relux.BatchSize(64),
        relux.Shuffle(true),
        relux.Verbose(true),
    )
    if err != nil {
        log.Fatal("Training failed:", err)
    }
    
    // Save trained model
    err = net.SaveFile("mnist_model.gob")
    if err != nil {
        log.Fatal("Failed to save model:", err)
    }
    
    // Evaluate on test set
    fmt.Println("Evaluating model...")
    predictions, err := net.PredictBatch(testX)
    if err != nil {
        log.Fatal("Prediction failed:", err)
    }
    
    // Calculate accuracy
    correct := 0
    for i, pred := range predictions {
        predClass := argmax(pred)
        trueClass := argmax(testY[i])
        if predClass == trueClass {
            correct++
        }
    }
    
    accuracy := float64(correct) / float64(len(testX)) * 100
    fmt.Printf("Test Accuracy: %.2f%% (%d/%d)\n", 
               accuracy, correct, len(testX))
    
    // Benchmark performance
    benchmark := net.Benchmark()
    fmt.Printf("Performance: %s using %s\n", 
               benchmark, benchmark.BackendInfo)
}

func argmax(slice []float64) int {
    maxIdx := 0
    for i := 1; i < len(slice); i++ {
        if slice[i] > slice[maxIdx] {
            maxIdx = i
        }
    }
    return maxIdx
}
```

---

## 📜 License & Compliance

**Apache License 2.0** — permissive open source with patent protection.

- ✅ Commercial use permitted
- ✅ Modification and distribution allowed  
- ✅ Patent grant included
- ✅ Liability limitations
- ✅ Compliance-friendly licensing

See [LICENSE](LICENSE) for complete terms.

---

## Support

### **Community Support** (Open Source)
- 🐛 **Issue Tracking**: [GitHub Issues](https://github.com/xDarkicex/relux/issues)
- 💬 **Technical Discussions**: [GitHub Discussions](https://github.com/xDarkicex/relux/discussions)
- 📚 **Documentation**: Comprehensive guides and API documentation
- 🚀 **Performance Issues**: Hardware acceleration troubleshooting

### **Commercial Inquiries**
- 📧 **Enterprise Licensing**: [gentry@xdarkicex.codes]
- 🔧 **Custom Development**: Specialized features and integrations
- 🎯 **Training & Support**: Go ML implementation consulting
- 🔒 **Security Assessments**: Compliance and security reviews

---

## **Acknowledgments**

### **Technical Foundation**
- **[rnxa Project](https://github.com/xDarkicex/rnxa)** – Hardware acceleration engine
- **Apple Metal Performance Shaders** – GPU acceleration infrastructure  
- **Go Team** — language design enabling systems-level ML

### **Historical Context**
- **Legacy neuron.go** – [8-year evolution](https://github.com/xDarkicex/GO-Portfolio/blob/master/app/neuron/neuron.go) from experimental code to research framework
- **Open Source ML Community** – Mathematical foundations and algorithmic insights

---

<div align="center">

**bf16-native transformer training & inference in Go**

*Powered by [rnxa](https://github.com/xDarkicex/rnxa) Hardware Acceleration*

⭐ **Star this repo if relux is useful to your research** ⭐

---

**Research-backed. Go-native. Hardware-accelerated.**

*© 2025 relux. Licensed under Apache 2.0.*

</div>