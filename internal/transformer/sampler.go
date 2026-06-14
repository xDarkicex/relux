package transformer

import (
	"math"
	"math/rand"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/relux/internal/alloc"
)

// Sampler produces token IDs from a logits vector. The
// standard triad is:
//
//	Greedy:      argmax(logits)
//	Temperature: divide logits by T before softmax; T->0
//	             approaches greedy, T->1 is the unmodified
//	             distribution, T>1 flattens (more random)
//	Top-k:       keep the top K logits, mask the rest to
//	             -inf before softmax
//	Top-p:       nucleus sampling — sort descending, keep
//	             the smallest prefix whose cumulative prob
//	             is >= p, mask the rest
//	Min-p:       filter tokens with prob < maxProb * minP
//
// Repetition control:
//
//	RepeatPenalty:    scale logits of seen tokens: >0 divide, <0 multiply
//	FrequencyPenalty: subtract freq * penalty from logits
//	PresencePenalty:  subtract penalty once if token appeared
//	RepeatWindow:     how many previous tokens to consider (0 = all)
//
// The fields compose: penalties → temperature → top-k → top-p → min-p → softmax → sample.
type Sampler struct {
	Temperature      float32
	TopK             int
	TopP             float32
	MinP             float32
	RepeatPenalty    float32
	FrequencyPenalty float32
	PresencePenalty  float32
	RepeatWindow     int
	Rand             *rand.Rand
}

// NewSampler constructs a Sampler with T=1, no top-k, no
// top-p (plain categorical).
func NewSampler() *Sampler {
	return &Sampler{Temperature: 1.0, TopP: 1.0}
}

// Greedy returns the argmax of logits.
func (s *Sampler) Greedy(logits []float32) int {
	if len(logits) == 0 {
		panic("Sampler.Greedy: empty logits")
	}
	bestIdx := 0
	bestVal := logits[0]
	for i, v := range logits[1:] {
		if v > bestVal {
			bestVal = v
			bestIdx = i + 1
		}
	}
	return bestIdx
}

var allocRandCounter uint64

// Sample returns one token ID sampled from the distribution
// over logits. previousTokens carries already-generated tokens
// for repetition control (nil = no penalty).
func (s *Sampler) Sample(logits []float32, previousTokens []int) int {
	n := len(logits)
	if n == 0 {
		panic("Sampler.Sample: empty logits")
	}

	// Alloc scratch once. defer-free for hot path; free at return.
	scratch := alloc.Float32(n)

	// 1. Copy logits → scratch.
	copy(scratch, logits)

	// 2. Repetition penalty. O(window) with uint16-bucketed map.
	s.applyRepetitionPenalties(scratch, previousTokens)

	// 3. Temperature.
	temp := s.Temperature
	if temp <= 0 {
		temp = 1e-5
	}
	if temp != 1.0 {
		for i := range scratch {
			scratch[i] /= temp
		}
	}

	// 4. Top-k filter.
	topK := s.TopK
	if topK < 0 {
		topK = 0
	}
	if topK >= n {
		topK = 0
	}
	if topK > 0 {
		threshold := kthLargest(scratch, topK)
		for i, v := range scratch {
			if v < threshold {
				scratch[i] = float32(math.Inf(-1))
			}
		}
	}

	// 5. Softmax into probs buffer.
	probs := alloc.Float32(n)
	maxV := scratch[0]
	for _, v := range scratch[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range scratch {
		if math.IsInf(float64(v), -1) {
			probs[i] = 0
		} else {
			probs[i] = fastexp32(v - maxV)
			sum += probs[i]
		}
	}
	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range probs {
			probs[i] *= invSum
		}
	}

	// 6. Top-p (nucleus).
	topP := s.TopP
	if topP <= 0 || topP > 1 {
		topP = 1.0
	}
	if topP < 1.0 {
		idx := argsortDescF32(probs)
		cum := float32(0)
		cutoff := n
		for i, k := range idx {
			cum += probs[k]
			if cum >= topP {
				cutoff = i + 1
				break
			}
		}
		for j := cutoff; j < n; j++ {
			probs[idx[j]] = 0
		}
		var newSum float32
		for _, v := range probs {
			newSum += v
		}
		if newSum > 0 {
			invSum := float32(1.0) / newSum
			for i := range probs {
				probs[i] *= invSum
			}
		}
	}

	// 7. Min-p filter.
	minP := s.MinP
	if minP > 0 {
		maxProb := float32(0)
		for _, p := range probs {
			if p > maxProb {
				maxProb = p
			}
		}
		threshold := maxProb * minP
		for i, p := range probs {
			if p < threshold {
				probs[i] = 0
			}
		}
		var newSum float32
		for _, p := range probs {
			newSum += p
		}
		if newSum > 0 {
			invSum := float32(1.0) / newSum
			for i := range probs {
				probs[i] *= invSum
			}
		}
	}

	// 8. Categorical sample.
	rng := s.Rand
	if rng == nil {
		seed := atomic.AddUint64(&allocRandCounter, 1) + uint64(time.Now().UnixNano())
		rng = rand.New(rand.NewSource(int64(seed)))
	}
	r := rng.Float32()
	var cum float32
	result := n - 1
	for i, v := range probs {
		cum += v
		if r < cum {
			result = i
			break
		}
	}

	alloc.Free(scratch)
	alloc.Free(probs)
	return result
}

// applyRepetitionPenalties modifies scratch in place.
// O(window) time, O(unique tokens in window) allocs via Go map
// (small — typical window=64, vocab is 32k-128k, dedup is ~64 entries).
func (s *Sampler) applyRepetitionPenalties(scratch []float32, prev []int) {
	if len(prev) == 0 {
		return
	}
	window := prev
	if s.RepeatWindow > 0 && len(prev) > s.RepeatWindow {
		window = prev[len(prev)-s.RepeatWindow:]
	}

	// Build seen set + frequency map in one pass.
	seen := make(map[int]bool, len(window))
	freq := make(map[int]int, len(window))
	for _, tok := range window {
		if tok >= 0 && tok < len(scratch) {
			seen[tok] = true
			freq[tok]++
		}
	}

	rp := s.RepeatPenalty
	fp := s.FrequencyPenalty
	pp := s.PresencePenalty

	for tok := range seen {
		// Repeat penalty: divide if positive, multiply if negative.
		if rp != 1.0 && rp != 0 {
			if scratch[tok] > 0 {
				scratch[tok] /= rp
			} else {
				scratch[tok] *= rp
			}
		}
		// Frequency penalty: subtract based on count.
		if fp != 0 {
			scratch[tok] -= fp * float32(freq[tok])
		}
		// Presence penalty: subtract once.
		if pp != 0 {
			scratch[tok] -= pp
		}
	}
}

// argsortDescF32 returns indices sorting probs descending. O(n²)
// selection sort; fine for vocab ≤ 50k.
func argsortDescF32(probs []float32) []int {
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < len(idx); i++ {
		maxJ := i
		for j := i + 1; j < len(idx); j++ {
			if probs[idx[j]] > probs[idx[maxJ]] {
				maxJ = j
			}
		}
		idx[i], idx[maxJ] = idx[maxJ], idx[i]
	}
	return idx
}

// kthLargest returns the k-th largest value via partial selection sort.
func kthLargest(arr []float32, k int) float32 {
	cp := alloc.Float32(len(arr))
	copy(cp, arr)
	for i := 0; i < k; i++ {
		maxJ := i
		for j := i + 1; j < len(cp); j++ {
			if cp[j] > cp[maxJ] {
				maxJ = j
			}
		}
		cp[i], cp[maxJ] = cp[maxJ], cp[i]
	}
	result := cp[k-1]
	alloc.Free(cp)
	return result
}
