package relux

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/xDarkicex/relux/dataset"
	"github.com/xDarkicex/relux/internal/compute"
	"github.com/xDarkicex/relux/internal/optim"
	"github.com/xDarkicex/relux/internal/transformer"
)

// Transformer is a full decoder-only language model. It is
// the new "be a real LLM" entry point alongside
// Network (the existing MLP framework).
//
// Construction:
//
//	t, _ := relux.NewTransformer(relux.ConfigTransformer{
//	    VocabSize:   100,
//	    DModel:      64,
//	    NumHeads:    4,
//	    NumKVHeads:  2,
//	    NumLayers:   2,
//	    DFF:         128,
//	    MaxSeqLen:   64,
//	    RopeBase:    10000,
//	    NormEps:     1e-5,
//	    Causal:      true,
//	})
//
// Training (single step):
//
//	loss, err := t.TrainStep([]int{1, 2, 3, 4}, []int{2, 3, 4, 5})
//
// Generation:
//
//	out, err := t.Generate([]int{1, 2}, 16, 0.8, 3)  // prompt, maxNew, temp, topK
//
// The Transformer composes:
//   - token embedding
//   - N transformer blocks (each = RMSNorm + MHA + RMSNorm + MLP)
//   - final RMSNorm
//   - lm_head (a Dense mapping dModel -> vocabSize)
//
// Mixed precision: master weights are float32 (Param.Data
// is float64 in the optim package; we cast to float32 on
// Forward and write the float32 result back to the active
// buffers; bfloat16 is a v2 follow-up). Gradients are
// float64 in the master; the active f32 path is for
// forward-pass matmul.
type Transformer struct {
	config ConfigTransformer

	// Layers
	embed    *transformer.Embedding
	blocks   []*transformer.Block
	finalNorm *transformer.RMSNorm
	lmHead   *transformer.Linear

	rope *transformer.RotaryEmbedding

	// Optimizer. Installed on the first Fit call (or via
	// SetOptimizerState after a Load). nil before either.
	adam       *optim.Adam
	optimState *optim.State

	// Backend is the compute backend (rnxa / MPS / Metal / CUDA
	// / pure Go). Set via SetBackend; nil means use the default
	// (pure Go, or whatever transformer.Backend is set to).
	backend compute.ComputeBackend
}

// ConfigTransformer holds the architectural hyperparameters.
type ConfigTransformer struct {
	VocabSize  int
	DModel     int
	NumHeads   int
	NumKVHeads int
	NumLayers  int
	DFF        int
	MaxSeqLen  int
	RopeBase   float32
	NormEps    float32
	Causal     bool

	// GradientCheckpointing enables activation recomputation
	// at block boundaries. During forward, intermediate
	// activations (attention scores, MLP hidden states) are
	// freed immediately; during backward, each block re-runs
	// its forward pass to regenerate them. Memory per block
	// drops to O(batch*seq*dModel) instead of O(seq²).
	GradientCheckpointing bool

	// FlashAttention enables the block-tiled Flash Attention 2
	// algorithm in MHA. Forward uses O(Br×Bc) memory instead of
	// O(seq²); backward recomputes attention weights on-the-fly.
	// Block sizes Br=Bc=64. Combine with GradientCheckpointing
	// for maximum memory savings.
	FlashAttention bool

	// FFNType selects the feedforward variant. "gelu" (default)
	// uses the standard 2-layer MLP with GELU. "swiglu" uses
	// SwiGLU: (SiLU(xW_up) ⊙ xW_gate) W_down — the standard
	// FFN in LLaMA 2/3, Mistral, and DeepSeek.
	FFNType string

	// AttnType selects the attention variant. "mha" (default)
	// uses standard multi-head attention. "mla" uses multi-head
	// latent attention with compressed KV cache.
	AttnType string

	// MLADimC is the KV compression dimension for MLA. Typically
	// 4× headDim (e.g., 512 for headDim=128). 0 means auto.
	MLADimC int

	// MLADimR is the per-head RoPE dimension for decoupled RoPE
	// in MLA. Typically headDim/2 (e.g., 64 for headDim=128).
	// 0 means auto. Must be even.
	MLADimR int
}

// FitConfig configures a training run via FitIteratorConfig.
type FitConfig struct {
	Steps int
	LR    float32
	RNG   *rand.Rand

	// Gradient accumulation: accumulate over K micro-batches
	// before one Adam step. 1 (default) = no accumulation.
	GradAccumSteps int

	// LR schedule. WarmupSteps sets the exact number of warmup
	// steps. 0 means auto: 10% of Steps if Steps >= 1000,
	// otherwise no warmup. MinLRRatio sets the cosine decay
	// floor as a fraction of peak LR (0 = default: 0.1).
	WarmupSteps int
	MinLRRatio  float32

	// Weight decay (AdamW, decoupled). 0 = none.
	WeightDecay float32

	// Early stopping based on validation loss. Stop after N
	// consecutive val checks without improvement. 0 = never.
	// Requires ValIterator and ValEvery to be set.
	EarlyStoppingPatience int

	// MetricsLogPath writes CSV training metrics to this file.
	// Empty string = no log.
	MetricsLogPath string

	// Gradient clipping max norm. 0 = default: 1.0.
	GradClipNorm float32

	// Checkpointing.
	CheckpointPath  string // base path, e.g. "ckpt.relv"
	CheckpointEvery int    // save every N optimizer steps (0 = never)

	// Validation.
	ValIterator dataset.Iterator
	ValEvery    int // validate every N optimizer steps (0 = never)

	// Callbacks.
	OnStep  func(step int, loss float32, ppl float32)
	OnEpoch func(epoch int, avgLoss float32, avgPPL float32)
	OnVal   func(step int, valLoss float32, valPPL float32)
}

// NewTransformer constructs a Transformer from a config. The
// weights are initialised with the same He / uniform patterns
// the underlying Modules use; the caller trains via
// TrainStep.
func NewTransformer(cfg ConfigTransformer) (*Transformer, error) {
	if cfg.VocabSize <= 0 || cfg.DModel <= 0 || cfg.NumHeads <= 0 ||
		cfg.NumKVHeads <= 0 || cfg.NumLayers <= 0 || cfg.DFF <= 0 ||
		cfg.MaxSeqLen <= 0 {
		return nil, errors.New("relux.NewTransformer: all config dims must be > 0")
	}
	if cfg.DModel%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("relux.NewTransformer: dModel=%d not divisible by numHeads=%d",
			cfg.DModel, cfg.NumHeads)
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("relux.NewTransformer: numHeads=%d not divisible by numKVHeads=%d",
			cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.RopeBase == 0 {
		cfg.RopeBase = 10000
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-5
	}

	headDim := cfg.DModel / cfg.NumHeads
	rope := transformer.NewRotaryEmbedding(headDim, cfg.RopeBase, cfg.MaxSeqLen)

	// Auto-detect the best compute backend. On macOS with rnxa
	// tag, this is MPS or Metal; on Linux with rnxa tag, CUDA;
	// otherwise pure Go. The backend is stored on the
	// Transformer and set as the global transformer.Backend
	// so all matmulBatched3D calls in the internal/transformer
	// package pick it up.
	backend := compute.NewComputeBackend()
	transformer.Backend = backend

	t := &Transformer{
		config:     cfg,
		embed:      transformer.NewEmbedding(cfg.VocabSize, cfg.DModel),
		finalNorm: transformer.NewRMSNorm(cfg.DModel, cfg.NormEps),
		rope:       rope,
		backend:    backend,
	}
	ffnType := transformer.FFNGELU
	if cfg.FFNType == "swiglu" {
		ffnType = transformer.FFNSwiGLU
	}
	attnType := transformer.ATTN_MHA
	dC := cfg.MLADimC
	dHR := cfg.MLADimR
	if cfg.AttnType == "mla" {
		attnType = transformer.ATTN_MLA
		headDim := cfg.DModel / cfg.NumHeads
		if dC <= 0 {
			dC = 4 * headDim
		}
		if dHR <= 0 {
			dHR = headDim / 2
		}
	}
	for i := 0; i < cfg.NumLayers; i++ {
		t.blocks = append(t.blocks,
			transformer.NewBlock(cfg.DModel, cfg.NumHeads, cfg.NumKVHeads, cfg.DFF, rope, cfg.Causal, ffnType, cfg.GradientCheckpointing, cfg.FlashAttention, attnType, dC, dHR))
	}
	// lmHead: a single linear mapping dModel -> vocabSize.
	// (We use the Linear primitive, not MLP, because MLP's
	// output is dModel, not vocabSize — the head needs to
	// produce logits over the full vocabulary.)
	t.lmHead = transformer.NewLinear(cfg.DModel, cfg.VocabSize)
	return t, nil
}

// Params returns all trainable parameters from the
// embedding, blocks, final norm, and lm head. Used by
// the optimizer to step.
func (t *Transformer) Params() []optim.Param {
	var out []optim.Param
	out = append(out, t.embed.Params()...)
	for _, b := range t.blocks {
		out = append(out, b.Params()...)
	}
	out = append(out, t.finalNorm.Params()...)
	out = append(out, t.lmHead.Params()...)
	return out
}

// Forward returns logits for batched token IDs. tokens is
// flat [batchSize, seqLen]; batchSize may be 1 for single-
// sequence inference. Output shape is [batchSize, seqLen,
// vocabSize].
//
// The Transformer preserves the current mode. Callers should
// call SetMode(Train) before training and SetMode(Inference)
// before generation.
func (t *Transformer) Forward(tokens []int, batchSize int) *transformer.Tensor {
	// 1. Embed → [batchSize, seqLen, dModel]
	h3D := t.embed.Forward(tokens, batchSize)
	// 2. Blocks
	for _, b := range t.blocks {
		h3D = b.Forward(h3D)
	}
	// 3. Final norm
	h3D = t.finalNorm.Forward(h3D)
	// 4. lmHead: shape [batchSize, seqLen, vocabSize]
	logits := t.lmHead.Forward(h3D)
	return logits
}

// setMode propagates the mode to all submodules.
func (t *Transformer) setMode(m transformer.Mode) {
	t.embed.SetMode(m)
	for _, b := range t.blocks {
		b.SetMode(m)
	}
	t.finalNorm.SetMode(m)
	t.lmHead.SetMode(m)
}

// SetMode is the public API to switch the Transformer
// between Train and Inference. Used by the autoregressive
// generation loop.
func (t *Transformer) SetMode(m transformer.Mode) {
	t.setMode(m)
}

// TrainStep runs a single training step on a single
// (input, target) pair. Returns the loss and populates
// the .Grad field of every trainable parameter. The
// Transformer is set to Train mode for the duration of
// the call.
//
// Loss is per-token cross-entropy: loss = -Σ log(softmax(logits_i)[target_i]).
// The gradient w.r.t. logits is softmax(logits) - onehot(target),
// which naturally focuses the gradient on the target position.
// This replaces the earlier MSE-on-logits approach which washed
// out the gradient signal across the full vocabulary.
//
// The full backward chain is:
//
//	gradOut -> lmHead.Backward
//	         -> finalNorm.Backward
//	         -> blocks[N-1].Backward -> ... -> blocks[0].Backward
//	         -> embed.Backward
//
// Each module's Backward returns its input gradient; the
// caller (Fit, or the user) is expected to step the
// optimizer afterwards.
func (t *Transformer) TrainStep(input, target []int, batchSize int) (float32, error) {
	if len(input) != len(target) {
		return 0, fmt.Errorf("relux.Transformer.TrainStep: input len %d != target len %d", len(input), len(target))
	}
	if len(input) == 0 {
		return 0, errors.New("relux.Transformer.TrainStep: empty input")
	}
	if len(input)%batchSize != 0 {
		return 0, fmt.Errorf("relux.Transformer.TrainStep: len(input)=%d not divisible by batchSize=%d", len(input), batchSize)
	}
	seq := len(input) / batchSize
	vocab := t.config.VocabSize
	t.setMode(transformer.Train)
	logits := t.Forward(input, batchSize)
	logitsData, _ := logits.ToF32()
	totalTokens := batchSize * seq
	gradOutData := make([]float32, totalTokens*vocab)
	loss := softmaxCELossGrad(logitsData, target, vocab, totalTokens, gradOutData)
	gradOut := transformer.NewTensor(gradOutData, batchSize, seq, vocab)

	// Full backward chain. Embed now outputs [batchSize, seq, dModel]
	// directly, so Backward receives the same 3D shape.
	gradOut = t.lmHead.Backward(gradOut)
	gradOut = t.finalNorm.Backward(gradOut)
	for i := len(t.blocks) - 1; i >= 0; i-- {
		gradOut = t.blocks[i].Backward(gradOut)
	}
	// gradOut is [batchSize, seq, dModel]; embed.Backward accepts 3D.
	t.embed.Backward(gradOut)
	return loss, nil
}

// EvalStep runs a forward pass and computes the cross-entropy
// loss without modifying gradients. Used for validation.
func (t *Transformer) EvalStep(input, target []int, batchSize int) float32 {
	t.setMode(transformer.Inference)
	logits := t.Forward(input, batchSize)
	logitsData, _ := logits.ToF32()
	totalTokens := batchSize * (len(input) / batchSize)
	return softmaxCELoss(logitsData, target, t.config.VocabSize, totalTokens)
}

// fastexp32 computes exp(x) using a pure-float32 bit-level
// approximation (Schraudolph, 1999). Accurate to ~2% relative
// error for |x| < 20. Much faster than float64 math.Exp in the
// hot cross-entropy loop because it avoids the f32→f64→f32
// conversion chain.
func fastexp32(x float32) float32 {
	if x < -87 {
		return 0
	}
	if x > 87 {
		return math.Float32frombits(0x7F800000) // +Inf
	}
	// exp(x) ≈ 2^(x * log2(e))
	// Multiply by 2^23, add IEEE 754 float32 bias, reinterpret.
	const factor float32 = float32(1 << 23) / 0.6931471805599453 // 2^23 / ln(2)
	const bias float32 = 127.0 * float32(1<<23) - 405000          // bias - adjustment
	return math.Float32frombits(uint32(int32(factor*x + bias)))
}

// softmaxCELossGrad computes cross-entropy loss and populates
// gradOut with dL/dlogits = softmax - onehot(target). Returns
// the sum of cross-entropy loss over all tokens.
func softmaxCELossGrad(logits []float32, target []int, vocab, totalTokens int, gradOut []float32) float32 {
	var loss float32
	for i := 0; i < totalTokens; i++ {
		tgt := target[i]
		base := i * vocab

		// Find max logit per token (numerical stability).
		maxVal := logits[base]
		for j := 1; j < vocab; j++ {
			if logits[base+j] > maxVal {
				maxVal = logits[base+j]
			}
		}

		// Single pass: exp + accumulate sum + store for grads.
		var sumExp float32
		for j := 0; j < vocab; j++ {
			diff := logits[base+j] - maxVal
			var e float32
			if diff < -20 {
				e = 0
			} else {
				e = fastexp32(diff)
			}
			gradOut[base+j] = e
			sumExp += e
		}
		invSum := float32(1.0) / sumExp

		// Loss: -log(softmax(target)).
		tgtProb := gradOut[base+tgt] * invSum
		if tgtProb > 1e-30 {
			loss += -float32(math.Log(float64(tgtProb)))
		} else {
			loss += 30 // cap for numerical safety
		}

		// Normalize: softmax = exp / sum(exp).
		for j := 0; j < vocab; j++ {
			gradOut[base+j] = gradOut[base+j] * invSum
		}
		// Subtract one-hot: dL/dlogits = softmax - onehot(target).
		gradOut[base+tgt] -= 1.0
	}
	return loss
}

// softmaxCELoss computes cross-entropy loss only (no gradient
// output). Used by EvalStep and validation loops.
func softmaxCELoss(logits []float32, target []int, vocab, totalTokens int) float32 {
	var loss float32
	for i := 0; i < totalTokens; i++ {
		tgt := target[i]
		base := i * vocab

		maxVal := logits[base]
		for j := 1; j < vocab; j++ {
			if logits[base+j] > maxVal {
				maxVal = logits[base+j]
			}
		}

		var sumExp float32
		var tgtExp float32
		for j := 0; j < vocab; j++ {
			diff := logits[base+j] - maxVal
			var e float32
			if diff < -20 {
				e = 0
			} else {
				e = fastexp32(diff)
			}
			if j == tgt {
				tgtExp = e
			}
			sumExp += e
		}
		prob := tgtExp / sumExp
		if prob > 1e-30 {
			loss += -float32(math.Log(float64(prob)))
		} else {
			loss += 30
		}
	}
	return loss
}

// Fit runs a small training loop for `steps` iterations
// on the given dataset. Each step picks a random
// sequence from dataset, runs TrainStep, accumulates
// gradients, and steps the optimizer. The optimizer is
// expected to be installed by the caller via the params'
// Grad fields. For v1 this is a thin convenience; the
// production Fit loop in relux.Network has more knobs
// (LR schedule, early stopping, etc.) which are out of
// scope for v1.
//
// `dataset` is a slice of token-ID sequences; each is a
// single training example. The `seqLen` is the number of
// tokens per training sample; longer sequences are
// truncated, shorter are padded with the pad token.
func (t *Transformer) Fit(dataset [][]int, seqLen int, steps int, lr float32, rng *rand.Rand) (float32, error) {
	if len(dataset) == 0 {
		return 0, errors.New("relux.Transformer.Fit: empty dataset")
	}
	if seqLen <= 0 {
		return 0, errors.New("relux.Transformer.Fit: seqLen must be > 0")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	// Adam optimizer with the standard transformer
	// learning rate (the caller passes lr; Adam's
	// defaults are 0.9/0.999/1e-8 for the betas/eps).
	if t.adam == nil {
		t.adam = &optim.Adam{
			LR:    lr,
			Beta1: 0.9,
			Beta2: 0.999,
			Eps:   1e-8,
		}
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	var totalLoss float32
	for step := 0; step < steps; step++ {
		// Pick a random example.
		ex := dataset[rng.Intn(len(dataset))]
		// Truncate or pad to seqLen+1 (input + target).
		if len(ex) < seqLen+1 {
			// Too short — skip this example; with a
			// real dataset we'd pad with the pad token.
			continue
		}
		// Random offset within the example.
		start := 0
		if len(ex) > seqLen+1 {
			start = rng.Intn(len(ex) - seqLen)
		}
		input := ex[start : start+seqLen]
		target := ex[start+1 : start+seqLen+1]
		// Zero the gradients.
		for _, p := range t.Params() {
			for i := range p.Grad {
				p.Grad[i] = 0
			}
		}
		// Forward + backward (TrainStep populates the
		// gradients through the full module chain:
		// lmHead -> finalNorm -> blocks -> embed).
		loss, err := t.TrainStep(input, target, 1)
		if err != nil {
			return totalLoss, err
		}
		// Optimizer step. Adam is the default; SGD is the
		// fallback. The ClipGradNorm prevents any one
		// batch from blowing up the params.
		optim.ClipGradNorm(t.Params(), 1.0)
		t.adam.Step(t.Params())
		totalLoss += loss
	}
	// Cache the optimizer state so Save() can include it.
	t.optimState = stateFromAdam(t.adam)
	return totalLoss / float32(steps), nil
}

// FitIterator trains the transformer using a streaming dataset
// iterator. It sequentially processes batches from the iterator,
// calling TrainStep on each input/target pair. When the iterator
// is exhausted and steps remain, Reset is called to begin another
// epoch.
//
// Learning rate follows a warmup + cosine decay schedule:
//
//   - warmup (first 10% of steps): linear ramp 0 → lr
//   - cosine decay (remaining 90%): lr → 0.1×lr
//
// Forward/backward is batched via matmulBatched3D (3D tensors).
// The loss computation (softmax + cross-entropy) runs as a per-token
// CPU loop and is the primary remaining bottleneck at small model sizes.
//
// If onStep is non-nil it is called after every optimizer step
// with (step, loss, perplexity).
func (t *Transformer) FitIterator(iter dataset.Iterator, steps int, lr float32, rng *rand.Rand, onStep func(step int, loss float32, ppl float32)) (float32, error) {
	return t.FitIteratorConfig(iter, FitConfig{
		Steps:  steps,
		LR:     lr,
		RNG:    rng,
		OnStep: onStep,
	})
}

// FitIteratorConfig trains the transformer using a streaming
// dataset iterator with full training-run support: gradient
// accumulation, periodic checkpointing, held-out validation,
// and epoch tracking.
func (t *Transformer) FitIteratorConfig(iter dataset.Iterator, cfg FitConfig) (float32, error) {
	if cfg.Steps <= 0 {
		return 0, errors.New("relux.Transformer.FitIteratorConfig: cfg.Steps must be > 0")
	}
	if cfg.RNG == nil {
		cfg.RNG = rand.New(rand.NewSource(1))
	}
	gradAccum := cfg.GradAccumSteps
	if gradAccum <= 0 {
		gradAccum = 1
	}
	checkpointEvery := cfg.CheckpointEvery
	valEvery := cfg.ValEvery

	peakLR := cfg.LR
	if t.adam == nil {
		t.adam = &optim.Adam{
			LR:          peakLR,
			Beta1:       0.9,
			Beta2:       0.999,
			Eps:         1e-8,
			WeightDecay: cfg.WeightDecay,
		}
	}
	t.adam.WeightDecay = cfg.WeightDecay
	// Resume from checkpoint: if optimState was set via
	// SetOptimizerState, load it into Adam now.
	if t.optimState != nil && t.adam != nil {
		t.adam.LoadState(*t.optimState)
		t.optimState = nil
	}

	// Warmup schedule.
	warmupSteps := cfg.WarmupSteps
	if warmupSteps == 0 && cfg.Steps >= 1000 {
		warmupSteps = cfg.Steps / 10
	}
	minLRRatio := cfg.MinLRRatio
	if minLRRatio <= 0 {
		minLRRatio = 0.1
	}
	gradClipNorm := cfg.GradClipNorm
	if gradClipNorm <= 0 {
		gradClipNorm = 1.0
	}
	var (
		totalLoss    float32
		epochLoss    float32
		epochSteps   int
		epoch        int
		validSteps   int
		microLoss    float32
		microCount   int
	)

	// Metrics log.
	var (
		metricsFile *os.File
		lastValLoss float32 = -1
	)
	if cfg.MetricsLogPath != "" {
		f, err := os.Create(cfg.MetricsLogPath)
		if err != nil {
			return 0, fmt.Errorf("metrics log: %w", err)
		}
		metricsFile = f
		defer f.Close()
		fmt.Fprintf(f, "step,train_loss,train_ppl,val_loss,val_ppl,lr\n")
	}

	// Early stopping state.
	var (
		bestValLoss    float32 = float32(math.Inf(1))
		noImproveCount int
		bestCkptPath   string
	)
	if cfg.EarlyStoppingPatience > 0 && cfg.CheckpointPath != "" {
		bestCkptPath = cfg.CheckpointPath + ".best.relv"
	}

	// checkpointPath inserts the step number before the extension.
	checkpointPath := func(base string, step int) string {
		for i := len(base) - 1; i >= 0; i-- {
			if base[i] == '.' {
				return base[:i] + fmt.Sprintf("_step_%d", step) + base[i:]
			}
		}
		return base + fmt.Sprintf("_step_%d", step)
	}

	// doVal runs a forward-only pass over the validation iterator.
	doVal := func(step int) {
		if cfg.ValIterator == nil {
			return
		}
		var valLoss float32
		var valTokens int
		cfg.ValIterator.Reset()
		for {
			batch, err := cfg.ValIterator.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				break
			}
			bSize := len(batch.Input)
			if bSize == 0 {
				continue
			}
			seqLen := len(batch.Input[0])
			flatInput := make([]int, bSize*seqLen)
			flatTarget := make([]int, bSize*seqLen)
			for bi := 0; bi < bSize; bi++ {
				copy(flatInput[bi*seqLen:], batch.Input[bi])
				copy(flatTarget[bi*seqLen:], batch.Target[bi])
			}
			valLoss += t.EvalStep(flatInput, flatTarget, bSize)
			valTokens += bSize * seqLen
		}
		if valTokens > 0 {
			valPPL := float32(math.Exp(float64(valLoss / float32(valTokens))))
			lastValLoss = valLoss / float32(valTokens)
			if cfg.OnVal != nil {
				cfg.OnVal(step, valLoss, valPPL)
			}
		}
	}

	getBatch := func() (dataset.Batch, error) {
		batch, err := iter.Next()
		if err == io.EOF {
			if epochSteps > 0 && cfg.OnEpoch != nil {
				avgLoss := float32(0.0)
				if epochSteps > 0 {
					avgLoss = epochLoss / float32(epochSteps)
				}
				avgPPL := float32(0.0)
				if avgLoss > 0 {
					avgPPL = float32(math.Exp(float64(avgLoss)))
				}
				cfg.OnEpoch(epoch, avgLoss, avgPPL)
			}
			epochLoss = 0
			epochSteps = 0
			epoch++
			iter.Reset()
			batch, err = iter.Next()
		}
		return batch, err
	}

	// The training loop is structured as: for each optimizer step,
	// accumulate over gradAccum micro-batches.
	scaledGradNorm := gradClipNorm / float32(gradAccum)
	for step := 0; step < cfg.Steps; step++ {
		// LR schedule (per optimizer step).
		if warmupSteps > 0 && step < warmupSteps {
			t.adam.LR = peakLR * float32(step+1) / float32(warmupSteps)
		} else if warmupSteps > 0 {
			progress := float32(step-warmupSteps) / float32(cfg.Steps-warmupSteps)
			t.adam.LR = peakLR * (minLRRatio + (1.0-minLRRatio)*0.5*(1.0+float32(math.Cos(float64(math.Pi*progress)))))
		}

		microLoss = 0
		microCount = 0
		microTokens := 0

		for acc := 0; acc < gradAccum; acc++ {
			batch, err := getBatch()
			if err != nil {
				return totalLoss / float32(max(validSteps, 1)), err
			}
			bSize := len(batch.Input)
			if bSize == 0 {
				continue
			}
			seqLen := len(batch.Input[0])
			flatInput := make([]int, bSize*seqLen)
			flatTarget := make([]int, bSize*seqLen)
			for bi := 0; bi < bSize; bi++ {
				copy(flatInput[bi*seqLen:], batch.Input[bi])
				copy(flatTarget[bi*seqLen:], batch.Target[bi])
			}

			// Zero gradients only at the start of an accumulation group.
			if acc == 0 {
				for _, p := range t.Params() {
					for j := range p.Grad {
						p.Grad[j] = 0
					}
				}
			}

			loss, err := t.TrainStep(flatInput, flatTarget, bSize)
			if err != nil {
				return totalLoss / float32(max(validSteps, 1)), err
			}
			microLoss += loss
			microCount++
			microTokens += bSize * seqLen
		}

		if microCount == 0 {
			continue
		}

		// Gradient clipping with accumulation scaling.
		optim.ClipGradNorm(t.Params(), scaledGradNorm)
		t.adam.Step(t.Params())

		totalLoss += microLoss
		epochLoss += microLoss
		epochSteps += microCount
		validSteps++

		// Checkpointing.
		if checkpointEvery > 0 && validSteps%checkpointEvery == 0 && cfg.CheckpointPath != "" {
			path := checkpointPath(cfg.CheckpointPath, validSteps)
			if err := t.SaveFile(path); err != nil {
				return totalLoss / float32(max(validSteps, 1)), fmt.Errorf("checkpoint save at step %d: %w", validSteps, err)
			}
		}

		// Validation.
		if valEvery > 0 && validSteps%valEvery == 0 {
			doVal(validSteps)
		}

		// Early stopping: track best val loss from the most recent val check.
		if cfg.EarlyStoppingPatience > 0 && valEvery > 0 && validSteps%valEvery == 0 {
			curValLoss := lastValLoss
			if curValLoss >= 0 && curValLoss < bestValLoss {
				bestValLoss = curValLoss
				noImproveCount = 0
				// Save best snapshot.
				if bestCkptPath != "" {
					if err := t.SaveFile(bestCkptPath); err != nil {
						return totalLoss / float32(max(validSteps, 1)), fmt.Errorf("best checkpoint save: %w", err)
					}
				}
			} else if curValLoss >= 0 {
				noImproveCount++
				if noImproveCount >= cfg.EarlyStoppingPatience {
					// Restore best snapshot.
					if bestCkptPath != "" {
						loaded, _, loadErr := LoadTransformerFile(bestCkptPath)
						if loadErr != nil {
							return totalLoss / float32(max(validSteps, 1)), fmt.Errorf("restore best checkpoint: %w", loadErr)
						}
						// Copy weights into current transformer.
						copyWeights(t, loaded)
					}
					break
				}
			}
		}

		// onStep callback.
		if cfg.OnStep != nil {
			ppl := float32(math.Exp(float64(microLoss / float32(microTokens))))
			cfg.OnStep(validSteps, microLoss, ppl)
		}

		// Metrics log.
		if metricsFile != nil {
			ppl := float32(math.Exp(float64(microLoss / float32(microTokens))))
			fmt.Fprintf(metricsFile, "%d,%f,%f,%f,%f,%f\n",
				validSteps, microLoss, ppl, lastValLoss, float32(math.Exp(float64(lastValLoss))), t.adam.LR)
		}
	}

	// Final epoch callback if we trained at all.
	if epochSteps > 0 && cfg.OnEpoch != nil {
		avgLoss := epochLoss / float32(epochSteps)
		avgPPL := float32(math.Exp(float64(avgLoss)))
		cfg.OnEpoch(epoch, avgLoss, avgPPL)
	}

	t.optimState = stateFromAdam(t.adam)
	return totalLoss / float32(max(validSteps, 1)), nil
}

// stateFromAdam extracts the State from an Adam.
func stateFromAdam(a *optim.Adam) *optim.State {
	if a == nil {
		return nil
	}
	st := a.State()
	return &st
}

// Generate runs the autoregressive loop with KV-cache.
// Returns up to maxNew tokens. The prompt is encoded as a
// sequence of token IDs; the output is the prompt plus the
// generated tokens.
//
// The first call creates a bf16 KV-cache wired into every
// block's MHA. Prefill computes K/V for the full prompt;
// subsequent decode steps process one token at a time with
// O(n) work per token instead of O(nÂ²).
//
// Temperature, topK, topP follow the Sampler semantics.
// `rng` is optional; if nil, a fresh source is used.
func (t *Transformer) Generate(prompt []int, maxNew int, temperature float32, topK int, rng *rand.Rand) ([]int, error) {
	if maxNew <= 0 {
		return nil, errors.New("relux.Transformer.Generate: maxNew must be > 0")
	}
	if len(prompt) == 0 {
		return nil, errors.New("relux.Transformer.Generate: empty prompt")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	sampler := &transformer.Sampler{
		Temperature: temperature,
		TopK:        topK,
		TopP:        1.0,
		Rand:        rng,
	}
	t.SetMode(transformer.Inference)

	// Create KV-cache and wire into blocks. For MLA, use the
	// compressed MLACache; for MHA, use the standard KVCache.
	if t.config.AttnType == "mla" {
		headDim := t.config.DModel / t.config.NumHeads
		dC := t.config.MLADimC
		if dC <= 0 {
			dC = 4 * headDim
		}
		dHR := t.config.MLADimR
		if dHR <= 0 {
			dHR = headDim / 2
		}
		mlaCache := transformer.NewMLACache(
			len(t.blocks), t.config.MaxSeqLen, dC, dHR,
		)
		for i, b := range t.blocks {
			mla := b.BlockMLA()
			mla.Cache = mlaCache
			mla.LayerIdx = i
		}
		defer func() {
			for _, b := range t.blocks {
				b.BlockMLA().Cache = nil
			}
			mlaCache.Reset()
		}()
	} else {
		cache := transformer.NewKVCacheSized(
			len(t.blocks), t.config.MaxSeqLen,
			t.config.NumKVHeads, t.config.DModel/t.config.NumHeads,
			transformer.BFloat16,
		)
		for i, b := range t.blocks {
			mha := b.BlockMHA()
			mha.Cache = cache
			mha.LayerIdx = i
		}
		defer func() {
			for _, b := range t.blocks {
				b.BlockMHA().Cache = nil
			}
			cache.Reset()
		}()
	}

	// Prefill: compute + cache K/V for the full prompt.
	prefillLogits := t.Forward(prompt, 1)
	prefillData, _ := prefillLogits.ToF32()
	vocab := prefillLogits.Shape()[2]
	lastPos := len(prompt) - 1
	next := sampler.Sample(prefillData[lastPos*vocab : (lastPos+1)*vocab], nil)
	out := append(append([]int{}, prompt...), next)

	// Decode loop: one token at a time from the cache.
	for i := 1; i < maxNew; i++ {
		logits := t.Forward([]int{next}, 1)
		logitsData, _ := logits.ToF32()
		next = sampler.Sample(logitsData[:vocab], out)
		out = append(out, next)
	}
	return out, nil
}

// GenerateResult is a single token from streaming generation.
type GenerateResult struct {
	Token   string // decoded token text
	TokenID int    // token ID
	Done    bool   // generation complete
	Reason  string // "eos", "max_tokens", or "stop_token"
}

// GenerateStream runs autoregressive generation and streams tokens
// through a channel. The channel is closed when generation completes.
// Callers must drain the channel to avoid goroutine leaks.
func (t *Transformer) GenerateStream(prompt []int, maxNew int, temperature float32, topK int, rng *rand.Rand) <-chan GenerateResult {
	ch := make(chan GenerateResult, 1)
	go func() {
		defer close(ch)
		tokens, _ := t.generateWithCallback(prompt, maxNew, temperature, topK, rng, func(res GenerateResult) bool {
			ch <- res
			return !res.Done
		})
		_ = tokens
	}()
	return ch
}

// generateWithCallback runs autoregressive generation, calling cb for
// each generated token. Used by both Generate and GenerateStream.
func (t *Transformer) generateWithCallback(prompt []int, maxNew int, temperature float32, topK int, rng *rand.Rand, cb func(GenerateResult) bool) ([]int, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	sampler := &transformer.Sampler{
		Temperature: temperature,
		TopK:        topK,
		TopP:        1.0,
		Rand:        rng,
	}
	t.SetMode(transformer.Inference)

	// Create KV-cache and wire into blocks.
	if t.config.AttnType == "mla" {
		headDim := t.config.DModel / t.config.NumHeads
		dC := t.config.MLADimC
		if dC <= 0 {
			dC = 4 * headDim
		}
		dHR := t.config.MLADimR
		if dHR <= 0 {
			dHR = headDim / 2
		}
		mlaCache := transformer.NewMLACache(len(t.blocks), t.config.MaxSeqLen, dC, dHR)
		for i, b := range t.blocks {
			mla := b.BlockMLA()
			mla.Cache = mlaCache
			mla.LayerIdx = i
		}
		defer func() {
			for _, b := range t.blocks {
				b.BlockMLA().Cache = nil
			}
			mlaCache.Reset()
		}()
	} else {
		cache := transformer.NewKVCacheSized(len(t.blocks), t.config.MaxSeqLen,
			t.config.NumKVHeads, t.config.DModel/t.config.NumHeads, transformer.BFloat16)
		for i, b := range t.blocks {
			mha := b.BlockMHA()
			mha.Cache = cache
			mha.LayerIdx = i
		}
		defer func() {
			for _, b := range t.blocks {
				b.BlockMHA().Cache = nil
			}
			cache.Reset()
		}()
	}

	// Prefill.
	prefillLogits := t.Forward(prompt, 1)
	prefillData, _ := prefillLogits.ToF32()
	vocab := prefillLogits.Shape()[2]
	lastPos := len(prompt) - 1

	next := sampler.Sample(prefillData[lastPos*vocab:(lastPos+1)*vocab], nil)
	out := append(append([]int{}, prompt...), next)

	eosID := -1
	for i := 1; i < maxNew; i++ {
		logits := t.Forward([]int{next}, 1)
		logitsData, _ := logits.ToF32()

		next = sampler.Sample(logitsData[:vocab], out)

		done := false
		reason := ""
		if next == eosID {
			done = true
			reason = "eos"
		} else if i >= maxNew-1 {
			done = true
			reason = "max_tokens"
		}

		tokenStr := fmt.Sprintf("[%d]", next)
		if !cb(GenerateResult{Token: tokenStr, TokenID: next, Done: done, Reason: reason}) {
			return out, nil
		}
		if done {
			return out, nil
		}
		out = append(out, next)
	}
	return out, nil
}

// ChatMessage is a single message in a conversation.
type ChatMessage struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// ChatTemplate formats messages into a model prompt.
type ChatTemplate interface {
	Apply(messages []ChatMessage) string
	Name() string
}

// ChatMLTemplate formats messages in ChatML format (OpenAI / DeepSeek).
// Format: <|im_start|>role\ncontent<|im_end|>\n
type ChatMLTemplate struct{}

func (c *ChatMLTemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}
func (c *ChatMLTemplate) Name() string { return "ChatML" }

// LLaMAChatTemplate formats messages in LLaMA chat format.
// Format: <s>[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST]
type LLaMAChatTemplate struct{ bosToken, eosToken string }

func NewLLaMAChatTemplate() *LLaMAChatTemplate {
	return &LLaMAChatTemplate{bosToken: "<s>", eosToken: "</s>"}
}
func (l *LLaMAChatTemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString(l.bosToken)
	var systemPrompt string
	var conversation []ChatMessage
	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
		} else {
			conversation = append(conversation, msg)
		}
	}
	for i, msg := range conversation {
		switch msg.Role {
		case "user":
			sb.WriteString("[INST] ")
			if i == 0 && systemPrompt != "" {
				sb.WriteString("<<SYS>>\n")
				sb.WriteString(systemPrompt)
				sb.WriteString("\n<</SYS>>\n\n")
			}
			sb.WriteString(msg.Content)
			sb.WriteString(" [/INST]")
		case "assistant":
			sb.WriteString(" ")
			sb.WriteString(msg.Content)
			sb.WriteString(l.eosToken)
			sb.WriteString(l.bosToken)
		}
	}
	return sb.String()
}
func (l *LLaMAChatTemplate) Name() string { return "LLaMA" }

// MistralChatTemplate formats messages in Mistral chat format.
type MistralChatTemplate struct{ bosToken, eosToken string }

func NewMistralChatTemplate() *MistralChatTemplate {
	return &MistralChatTemplate{bosToken: "<s>", eosToken: "</s>"}
}
func (m *MistralChatTemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder
	sb.WriteString(m.bosToken)
	for i, msg := range messages {
		switch msg.Role {
		case "user":
			sb.WriteString("[INST] ")
			sb.WriteString(msg.Content)
			sb.WriteString(" [/INST]")
		case "assistant":
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(msg.Content)
			sb.WriteString(m.eosToken)
			if i < len(messages)-1 {
				sb.WriteString(m.bosToken)
			}
		case "system":
			if i == 0 {
				sb.WriteString("[INST] ")
				sb.WriteString(msg.Content)
				sb.WriteString(" [/INST]")
			}
		}
	}
	return sb.String()
}
func (m *MistralChatTemplate) Name() string { return "Mistral" }

// Config returns the transformer's config. Useful for
// serialization / debug.
func (t *Transformer) Config() ConfigTransformer { return t.config }

// AdamLR returns the current Adam learning rate. Returns 0 if
// no Adam is installed. Exposed for tests and diagnostics.
func (t *Transformer) AdamLR() float32 {
	if t.adam == nil {
		return 0
	}
	return t.adam.LR
}

// SetBackend replaces the compute backend. Set to nil to use
// pure Go (the default). The global transformer.Backend is
// also updated so the internal matmulBatched3D helper picks
// it up.
func (t *Transformer) SetBackend(b compute.ComputeBackend) {
	t.backend = b
	transformer.Backend = b
}

// Close releases the Transformer's compute backend. Safe to
// call multiple times (nil backend is a no-op).
func (t *Transformer) Close() error {
	if t.backend != nil {
		return t.backend.Close()
	}
	return nil
}

// GetEmbedding returns the embedding module. Used by the v1
// serializer to walk the layers.
func (t *Transformer) GetEmbedding() *transformer.Embedding { return t.embed }

// GetBlocks returns the slice of transformer blocks.
func (t *Transformer) GetBlocks() []*transformer.Block { return t.blocks }

// GetFinalNorm returns the final RMSNorm.
func (t *Transformer) GetFinalNorm() *transformer.RMSNorm { return t.finalNorm }

// GetLMHead returns the language model head.
func (t *Transformer) GetLMHead() *transformer.Linear { return t.lmHead }

// GetRoPE returns the rotary embedding module.
func (t *Transformer) GetRoPE() *transformer.RotaryEmbedding { return t.rope }

// SetRoPE replaces the rotary embedding module. Used by the
// v1 deserializer to swap in a freshly-built RoPE (the
// constructor in NewTransformer builds its own).
func (t *Transformer) SetRoPE(r *transformer.RotaryEmbedding) { t.rope = r }

// Summary returns a one-line architecture summary.
func (t *Transformer) Summary() string {
	return fmt.Sprintf("Transformer(vocab=%d, dModel=%d, heads=%d, kvHeads=%d, layers=%d, dFF=%d, maxSeqLen=%d)",
		t.config.VocabSize, t.config.DModel, t.config.NumHeads, t.config.NumKVHeads,
		t.config.NumLayers, t.config.DFF, t.config.MaxSeqLen)
}

// copyWeights copies all parameter data from src to dst.
func copyWeights(dst, src *Transformer) {
	dstParams := dst.Params()
	srcParams := src.Params()
	for i := range dstParams {
		copy(dstParams[i].Data, srcParams[i].Data)
	}
}
