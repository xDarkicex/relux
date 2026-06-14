package relux

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"

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
	for i := 0; i < cfg.NumLayers; i++ {
		t.blocks = append(t.blocks,
			transformer.NewBlock(cfg.DModel, cfg.NumHeads, cfg.NumKVHeads, cfg.DFF, rope, cfg.Causal))
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

// Forward returns the logits for a single training step
// with the given input token IDs. The output shape is
// [seq, vocabSize] in float32.
//
// The Transformer preserves the current mode. Callers
// should call SetMode(Train) before training and
// SetMode(Inference) before generation.
func (t *Transformer) Forward(tokens []int) *transformer.Tensor {
	// 1. Embed
	h := t.embed.Forward(tokens)
	// h is shape [seq, dModel]. MHA/Block want [batch,
	// seq, dModel] with batch=1.
	h3D := h.Reshape(1, len(tokens), t.config.DModel)
	// 2. Blocks
	for _, b := range t.blocks {
		h3D = b.Forward(h3D)
	}
	// 3. Final norm
	h3D = t.finalNorm.Forward(h3D)
	// 4. lmHead: shape [1, seq, vocabSize]
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
func (t *Transformer) TrainStep(input, target []int) (float32, error) {
	if len(input) != len(target) {
		return 0, fmt.Errorf("relux.Transformer.TrainStep: input len %d != target len %d", len(input), len(target))
	}
	if len(input) == 0 {
		return 0, errors.New("relux.Transformer.TrainStep: empty input")
	}
	t.setMode(transformer.Train)
	logits := t.Forward(input)
	logitsData, _ := logits.ToF32()
	seq := len(input)
	vocab := t.config.VocabSize
	var loss float32
	gradOutData := make([]float32, seq*vocab)
	for i := 0; i < seq; i++ {
		tgt := target[i]
		if tgt < 0 || tgt >= vocab {
			return 0, fmt.Errorf("relux.Transformer.TrainStep: target[%d]=%d out of vocab range", i, tgt)
		}
		base := i * vocab
		// Log-sum-exp: subtract max for numerical stability.
		maxVal := logitsData[base]
		for j := 1; j < vocab; j++ {
			if logitsData[base+j] > maxVal {
				maxVal = logitsData[base+j]
			}
		}
		// Softmax + CE gradient in one pass.
		var sumExp float32
		for j := 0; j < vocab; j++ {
			e := float32(math.Exp(float64(logitsData[base+j] - maxVal)))
			gradOutData[base+j] = e
			sumExp += e
		}
		invSum := 1.0 / sumExp
		loss += -float32(math.Log(float64(gradOutData[base+tgt] * invSum)))
		for j := 0; j < vocab; j++ {
			gradOutData[base+j] = gradOutData[base+j] * invSum
		}
		gradOutData[base+tgt] -= 1.0
	}
	gradOut := transformer.NewTensor(gradOutData, 1, seq, vocab)

	// Full backward chain. Each module's Backward returns
	// the gradient w.r.t. its input; we thread that into
	// the next module's Backward. The embed was called
	// with a 2D [seq, dModel] input (see Forward above),
	// so we reshape the 3D gradOut from the first block
	// back to 2D before passing to embed.Backward.
	gradOut = t.lmHead.Backward(gradOut)
	gradOut = t.finalNorm.Backward(gradOut)
	for i := len(t.blocks) - 1; i >= 0; i-- {
		gradOut = t.blocks[i].Backward(gradOut)
	}
	// gradOut is [1, seq, dModel]; squeeze the batch dim.
	gradOut2D := gradOut.Reshape(seq, t.config.DModel)
	t.embed.Backward(gradOut2D) // returns nil (no input grad)
	return loss, nil
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
		loss, err := t.TrainStep(input, target)
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
// Batches are processed one sequence at a time through TrainStep
// (no batched matmul). Batched forward/backward is deferred to a
// dedicated follow-up project.
//
// If onStep is non-nil it is called after every optimizer step
// with (step, loss, perplexity).
func (t *Transformer) FitIterator(iter dataset.Iterator, steps int, lr float32, rng *rand.Rand, onStep func(step int, loss float32, ppl float32)) (float32, error) {
	if steps <= 0 {
		return 0, errors.New("relux.Transformer.FitIterator: steps must be > 0")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	peakLR := lr
	if t.adam == nil {
		t.adam = &optim.Adam{
			LR:    peakLR,
			Beta1: 0.9,
			Beta2: 0.999,
			Eps:   1e-8,
		}
	}
	// Warmup: 10% of total steps. For runs shorter than
	// 1000 steps the warmup ramp would consume too much
	// of the training budget relative to the tiny model
	// size, so we use constant LR.
	warmupSteps := 0
	if steps >= 1000 {
		warmupSteps = steps / 10
	}
	var totalLoss float32
	validSteps := 0
	for step := 0; step < steps; step++ {
		// LR schedule.
		if warmupSteps > 0 && step < warmupSteps {
			// Linear warmup: 0 → peakLR.
			t.adam.LR = peakLR * float32(step+1) / float32(warmupSteps)
		} else if warmupSteps > 0 {
			// Cosine decay: peakLR → 0.1×peakLR.
			progress := float32(step-warmupSteps) / float32(steps-warmupSteps)
			t.adam.LR = peakLR * (0.1 + 0.45*(1.0+float32(math.Cos(float64(math.Pi*progress)))))
		}
		batch, err := iter.Next()
		if err == io.EOF {
			iter.Reset()
			batch, err = iter.Next()
			if err != nil {
				continue
			}
		} else if err != nil {
			return totalLoss / float32(max(validSteps, 1)), err
		}
		// Process each sequence in the batch one at a time.
		for i := range batch.Input {
			input := batch.Input[i]
			target := batch.Target[i]
			// Zero gradients.
			for _, p := range t.Params() {
				for j := range p.Grad {
					p.Grad[j] = 0
				}
			}
			loss, err := t.TrainStep(input, target)
			if err != nil {
				return totalLoss / float32(max(validSteps, 1)), err
			}
			optim.ClipGradNorm(t.Params(), 1.0)
			t.adam.Step(t.Params())
			totalLoss += loss
			validSteps++
			if onStep != nil {
				ppl := float32(math.Exp(float64(loss / float32(len(input)))))
				onStep(validSteps, loss, ppl)
			}
		}
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

	// Create KV-cache (bf16, pre-allocated for MaxSeqLen) and wire into blocks.
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

	// Prefill: compute + cache K/V for the full prompt.
	prefillLogits := t.Forward(prompt)
	prefillData, _ := prefillLogits.ToF32()
	vocab := prefillLogits.Shape()[2]
	lastPos := len(prompt) - 1
	next := sampler.Sample(prefillData[lastPos*vocab : (lastPos+1)*vocab])
	out := append(append([]int{}, prompt...), next)

	// Decode loop: one token at a time from the cache.
	for i := 1; i < maxNew; i++ {
		logits := t.Forward([]int{next})
		logitsData, _ := logits.ToF32()
		next = sampler.Sample(logitsData[:vocab])
		out = append(out, next)
	}
	return out, nil
}

// Config returns the transformer's config. Useful for
// serialization / debug.
func (t *Transformer) Config() ConfigTransformer { return t.config }

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
