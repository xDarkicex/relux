package relux

import (
	"fmt"
	"math/rand"

	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/transformer"
)

// SpeculativeConfig configures speculative decoding.
type SpeculativeConfig struct {
	// NumSpeculate is the number of tokens the draft model speculates
	// ahead before the target model verifies (default: 5).
	NumSpeculate int

	// Sampling configures token selection for both models.
	Sampling transformer.Sampler
}

// DefaultSpeculativeConfig returns sensible defaults.
func DefaultSpeculativeConfig() SpeculativeConfig {
	return SpeculativeConfig{
		NumSpeculate: 5,
		Sampling:     *transformer.NewSampler(),
	}
}

// SpeculativeDecoder accelerates autoregressive generation using a
// small draft model that speculates multiple tokens ahead, verified
// in parallel by a large target model.
//
// Algorithm (modified rejection sampling, Leviathan et al. 2023):
//  1. Draft model generates K tokens sequentially (fast, O(K × d_draft²))
//  2. Target model verifies all K tokens in one parallel forward pass
//     (O((K+1) × d_target²))
//  3. Accept draft token i if r < min(1, p_target[i] / p_draft[i])
//  4. On first rejection at position j: resample from
//     norm(max(0, p_target[j] - p_draft[j])), discard tail
//  5. Continue from position j+1
//
// This provides 2-4× speedup when the draft model is substantially
// smaller than the target and the draft has high agreement.
//
// All internal buffers are alloc.Float32 (off-heap mmap'd, out of Go GC).
// Per-iteration memory is O(K × vocab + K × dModel) with alloc reuse.
type SpeculativeDecoder struct {
	draft  *Transformer
	target *Transformer

	draftCache  *transformer.KVCache
	targetCache *transformer.KVCache

	config SpeculativeConfig
	rng    *rand.Rand

	// Pre-allocated buffer pool (reused across iterations).
	bufSize int // max(vocab * numSpeculate, vocab)
}

// NewSpeculativeDecoder creates a speculative decoder. Both models
// must share the same vocabulary size. The draft should be smaller
// (fewer layers / smaller dModel) for speedup.
func NewSpeculativeDecoder(draft, target *Transformer, config SpeculativeConfig) (*SpeculativeDecoder, error) {
	if config.NumSpeculate <= 0 {
		config.NumSpeculate = 5
	}
	if draft.Config().VocabSize != target.Config().VocabSize {
		return nil, fmt.Errorf("speculative: vocab mismatch: draft=%d target=%d",
			draft.Config().VocabSize, target.Config().VocabSize)
	}
	rng := config.Sampling.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	return &SpeculativeDecoder{
		draft:  draft,
		target: target,
		config: config,
		rng:    rng,
	}, nil
}

// Generate runs speculative decoding. Returns generated token IDs
// and the draft acceptance rate (0-1).
//
// Both models must be in Inference mode. The caller owns the
// returned slice; it is Go-heap allocated.
func (sd *SpeculativeDecoder) Generate(prompt []int, maxNew int) ([]int, float32, error) {
	if len(prompt) == 0 {
		return nil, 0, fmt.Errorf("speculative: empty prompt")
	}

	vocab := sd.draft.Config().VocabSize
	K := sd.config.NumSpeculate
	batchSize := 1

	// Allocate KV caches.
	sd.draftCache = transformer.NewKVCacheSized(
		len(sd.draft.GetBlocks()), sd.draft.Config().MaxSeqLen,
		sd.draft.Config().NumKVHeads, sd.draft.Config().DModel/sd.draft.Config().NumHeads,
		transformer.BFloat16,
	)
	sd.targetCache = transformer.NewKVCacheSized(
		len(sd.target.GetBlocks()), sd.target.Config().MaxSeqLen,
		sd.target.Config().NumKVHeads, sd.target.Config().DModel/sd.target.Config().NumHeads,
		transformer.BFloat16,
	)
	defer func() {
		sd.clearCaches()
		sd.draftCache.Reset()
		sd.targetCache.Reset()
	}()

	sd.wireCaches(sd.draft, sd.draftCache)
	sd.wireCaches(sd.target, sd.targetCache)

	// Prefill: both models process the full prompt.
	sd.draft.SetMode(transformer.Inference)
	sd.target.SetMode(transformer.Inference)
	_ = sd.draft.Forward(prompt, batchSize)
	_ = sd.target.Forward(prompt, batchSize)

	generated := make([]int, 0, maxNew)
	prev := append([]int{}, prompt...)
	totalDrafted := 0
	totalAccepted := 0
	startPos := len(prompt)

	for len(generated) < maxNew {
		remaining := maxNew - len(generated)
		k := K
		if k > remaining {
			k = remaining
		}

		// Phase 1: Draft generates k tokens.
		draftTokens, draftLogits := sd.draftPhase(prev, startPos, k, vocab)

		// Phase 2: Target verifies all k tokens in one forward pass.
		targetLogits := sd.verifyPhase(prev, draftTokens, startPos, vocab)

		// Phase 3: Accept/reject.
		nAccepted, resampled := sd.acceptPhase(draftTokens, draftLogits, targetLogits, vocab)

		for i := 0; i < nAccepted; i++ {
			generated = append(generated, draftTokens[i])
			prev = append(prev, draftTokens[i])
		}
		generated = append(generated, resampled)
		prev = append(prev, resampled)

		totalDrafted += k
		totalAccepted += nAccepted
		startPos += nAccepted + 1
	}

	rate := float32(0)
	if totalDrafted > 0 {
		rate = float32(totalAccepted) / float32(totalDrafted)
	}
	return generated[:maxNew], rate, nil
}

// draftPhase runs the small draft model sequentially for k steps.
func (sd *SpeculativeDecoder) draftPhase(
	prev []int, startPos, k, vocab int,
) (tokens []int, logits [][]float32) {
	tokens = make([]int, 0, k)
	logits = allocLogitMatrix(k, vocab) // alloc.Float32 backed

	lastToken := prev[len(prev)-1]
	for i := 0; i < k; i++ {
		out := sd.draft.Forward([]int{lastToken}, 1)
		outData, _ := out.ToF32()

		copy(logits[i], outData[:vocab])
		next := sd.config.Sampling.Sample(outData[:vocab], prev)

		tokens = append(tokens, next)
		prev = append(prev, next)
		lastToken = next
	}
	return
}

// verifyPhase runs the target model on [last_input] + draft_tokens
// in a single batched forward pass. Returns logits for all positions.
func (sd *SpeculativeDecoder) verifyPhase(
	prev []int, draftTokens []int, startPos, vocab int,
) [][]float32 {
	// Build input: last prompt token + all draft tokens.
	nTokens := 1 + len(draftTokens)
	input := make([]int, nTokens)
	input[0] = prev[len(prev)-1]
	copy(input[1:], draftTokens)

	out := sd.target.Forward(input, 1)
	outData, _ := out.ToF32()

	nOut := len(outData) / vocab
	logits := allocLogitMatrix(nOut, vocab)
	for i := 0; i < nOut; i++ {
		copy(logits[i], outData[i*vocab:(i+1)*vocab])
	}
	return logits
}

// acceptPhase runs modified rejection sampling over the k draft tokens.
// Returns (numAccepted, resampledToken).
func (sd *SpeculativeDecoder) acceptPhase(
	draftTokens []int,
	draftLogits, targetLogits [][]float32,
	vocab int,
) (int, int) {
	for i, token := range draftTokens {
		dProbs := softmaxAlloc(draftLogits[i])
		tProbs := softmaxAlloc(targetLogits[i])

		dP := dProbs[token]
		tP := tProbs[token]

		acceptProb := tP / max32(dP, 1e-10)
		if acceptProb > 1 {
			acceptProb = 1
		}

		if sd.rng.Float32() < acceptProb {
			alloc.Free(dProbs)
			alloc.Free(tProbs)
			continue // accepted
		}

		// Rejected at position i. Resample from norm(max(0, t - d)).
		resampled := sd.resampleRejected(dProbs, tProbs)
		alloc.Free(dProbs)
		alloc.Free(tProbs)

		// Free remaining logits.
		freeLogitMatrix(draftLogits)
		freeLogitMatrix(targetLogits)
		return i, resampled
	}

	// All k accepted. Sample next from target's k-th position.
	lastLogits := targetLogits[len(draftTokens)]
	next := sd.config.Sampling.Sample(lastLogits, nil)
	freeLogitMatrix(draftLogits)
	freeLogitMatrix(targetLogits)
	return len(draftTokens), next
}

// resampleRejected samples from norm(max(0, p_target - p_draft)).
func (sd *SpeculativeDecoder) resampleRejected(dProbs, tProbs []float32) int {
	n := len(tProbs)
	adjusted := alloc.Float32(n)

	var sum float32
	for i := range adjusted {
		diff := tProbs[i] - dProbs[i]
		if diff < 0 {
			diff = 0
		}
		adjusted[i] = diff
		sum += diff
	}

	var result int
	if sum > 0 {
		invSum := float32(1.0) / sum
		r := sd.rng.Float32()
		var cum float32
		for i := range adjusted {
			cum += adjusted[i] * invSum
			if r < cum {
				result = i
				break
			}
		}
	} else {
		// Fallback: sample from target distribution.
		r := sd.rng.Float32()
		var cum float32
		for i := range tProbs {
			cum += tProbs[i]
			if r < cum {
				result = i
				break
			}
		}
	}

	alloc.Free(adjusted)
	return result
}

// wireCaches installs KV caches on all blocks.
func (sd *SpeculativeDecoder) wireCaches(t *Transformer, cache *transformer.KVCache) {
	for i, b := range t.GetBlocks() {
		mha := b.BlockMHA()
		mha.Cache = cache
		mha.LayerIdx = i
	}
}

// clearCaches removes KV caches from all blocks.
func (sd *SpeculativeDecoder) clearCaches() {
	sd.unwireCaches(sd.draft)
	sd.unwireCaches(sd.target)
	sd.draftCache = nil
	sd.targetCache = nil
}

func (sd *SpeculativeDecoder) unwireCaches(t *Transformer) {
	for _, b := range t.GetBlocks() {
		b.BlockMHA().Cache = nil
	}
}

// --- alloc-backed helpers ---

// allocLogitMatrix allocates rows × cols float32 via alloc.Float32.
// Each row is independently allocated for freeLogitMatrix to free.
func allocLogitMatrix(rows, cols int) [][]float32 {
	if rows == 0 {
		return nil
	}
	m := make([][]float32, rows)
	for i := range m {
		m[i] = alloc.Float32(cols)
	}
	return m
}

// freeLogitMatrix frees all rows and the outer slice.
func freeLogitMatrix(m [][]float32) {
	for i := range m {
		if m[i] != nil {
			alloc.Free(m[i])
			m[i] = nil
		}
	}
}

// softmaxAlloc computes softmax in an alloc.Float32 buffer.
func softmaxAlloc(logits []float32) []float32 {
	n := len(logits)
	probs := alloc.Float32(n)

	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range logits {
		probs[i] = fastexp32(v - maxV)
		sum += probs[i]
	}
	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range probs {
			probs[i] *= invSum
		}
	}
	return probs
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
