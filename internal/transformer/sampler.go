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
//   Greedy:      argmax(logits)
//   Temperature: divide logits by T before softmax; T->0
//                approaches greedy, T->1 is the unmodified
//                distribution, T>1 flattens (more random)
//   Top-k:       keep the top K logits, mask the rest to
//                -inf before softmax
//   Top-p:       nucleus sampling — sort descending, keep
//                the smallest prefix whose cumulative prob
//                is >= p, mask the rest
//
// The fields compose: temperature is applied first, then
// top-k, then top-p, then softmax, then categorical
// sampling. With T=1, K=0, P=1, the sampler is the plain
// categorical.
type Sampler struct {
	Temperature float32
	TopK        int
	TopP        float32
	Rand        *rand.Rand // optional; if nil, uses a fresh source per call
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

// allocRandCounter is an atomic counter used as a seed
// for the default RNG so multiple goroutines don't collide.
var allocRandCounter uint64

// Sample returns one token ID sampled from the distribution
// over logits, applying Temperature, TopK, TopP. The returned
// integer is in [0, len(logits)). The logits slice is not
// modified.
func (s *Sampler) Sample(logits []float32) int {
	if len(logits) == 0 {
		panic("Sampler.Sample: empty logits")
	}
	temperature := s.Temperature
	if temperature <= 0 {
		temperature = 1e-5 // T->0 -> greedy
	}
	topK := s.TopK
	if topK < 0 {
		topK = 0
	}
	if topK > len(logits) {
		topK = len(logits)
	}
	topP := s.TopP
	if topP <= 0 || topP > 1 {
		topP = 1.0
	}

	// Step 1: temperature scaling. Copy logits into a
	// scratch buffer (the input is not modified).
	scratch := alloc.Float32(len(logits))
	defer alloc.Free(scratch)
	for i, v := range logits {
		scratch[i] = v / temperature
	}

	// Step 2: top-k. Find the k-th largest value; mask
	// everything below to -inf.
	if topK > 0 && topK < len(logits) {
		kthVal := kthLargest(scratch, topK)
		for i, v := range scratch {
			if v < kthVal {
				scratch[i] = float32(math.Inf(-1))
			}
		}
	}

	// Step 3: softmax into a second scratch.
	probs := alloc.Float32(len(logits))
	defer alloc.Free(probs)
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
			probs[i] = float32(math.Exp(float64(v - maxV)))
			sum += probs[i]
		}
	}
	if sum > 0 {
		invSum := 1.0 / sum
		for i := range probs {
			probs[i] *= invSum
		}
	}

	// Step 4: top-p. Sort indices by descending prob;
	// compute cumulative sum; find the cutoff.
	if topP < 1.0 {
		idx := argsortDescF32(probs)
		cum := float32(0)
		cutoff := len(probs)
		for i, k := range idx {
			cum += probs[k]
			if cum >= topP {
				cutoff = i + 1
				break
			}
		}
		for j := cutoff; j < len(probs); j++ {
			probs[idx[j]] = 0
		}
		var newSum float32
		for _, v := range probs {
			newSum += v
		}
		if newSum > 0 {
			invSum := 1.0 / newSum
			for i := range probs {
				probs[i] *= invSum
			}
		}
	}

	// Step 5: categorical sample.
	rng := s.Rand
	if rng == nil {
		seed := atomic.AddUint64(&allocRandCounter, 1) + uint64(time.Now().UnixNano())
		rng = rand.New(rand.NewSource(int64(seed)))
	}
	r := rng.Float32()
	var cum float32
	for i, v := range probs {
		cum += v
		if r < cum {
			return i
		}
	}
	return len(probs) - 1 // numerical fallback
}

// argsortDescF32 returns the indices that would sort probs
// in descending order. Uses selection sort (O(n^2)); fine
// for vocab sizes up to ~50k.
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

// kthLargest returns the k-th largest value in arr. Uses
// a partial selection sort.
func kthLargest(arr []float32, k int) float32 {
	cp := alloc.Float32(len(arr))
	defer alloc.Free(cp)
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
	return cp[k-1]
}
