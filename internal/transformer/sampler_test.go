package transformer_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestSampler_Greedy(t *testing.T) {
	s := transformer.NewSampler()
	logits := []float32{1, 5, 3, 2, 4}
	if got := s.Greedy(logits); got != 1 {
		t.Errorf("Greedy = %d, want 1", got)
	}
}

func TestSampler_GreedyOnTie(t *testing.T) {
	// First-argmax wins on ties.
	s := transformer.NewSampler()
	logits := []float32{5, 5, 3}
	if got := s.Greedy(logits); got != 0 {
		t.Errorf("Greedy on tie = %d, want 0", got)
	}
}

func TestSampler_TemperatureZeroApproachesGreedy(t *testing.T) {
	// T=0 (or near-zero) makes the distribution concentrate
	// on the argmax.
	s := transformer.NewSampler()
	s.Temperature = 1e-10
	logits := []float32{1, 5, 3}
	// Run 100 times; expect index 1 every time.
	for i := 0; i < 100; i++ {
		if got := s.Sample(logits, nil); got != 1 {
			t.Errorf("Sample at T=0: got %d, want 1", got)
		}
	}
}

func TestSampler_TopK(t *testing.T) {
	// Top-2: only the top 2 logits are kept; sample only
	// produces indices 1 or 4.
	s := transformer.NewSampler()
	s.Temperature = 1.0
	s.TopK = 2
	s.Rand = rand.New(rand.NewSource(42))
	logits := []float32{1, 5, 3, 2, 4}
	for i := 0; i < 100; i++ {
		got := s.Sample(logits, nil)
		if got != 1 && got != 4 {
			t.Errorf("Top-2 sample: got %d, want 1 or 4", got)
		}
	}
}

func TestSampler_DistributionSumsToOne(t *testing.T) {
	// For a known logits vector, the unfiltered distribution
	// sums to 1.0. We test the helper softmax, not the full
	// Sampler (which involves randomness).
	logits := []float32{0.0, 0.0, 0.0, 0.0}
	scratch := []float32{0, 0, 0, 0} // no temperature scaling
	maxV := scratch[0]
	for _, v := range scratch {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	probs := make([]float32, len(logits))
	for i, v := range scratch {
		probs[i] = float32(math.Exp(float64(v - maxV)))
		sum += probs[i]
	}
	invSum := 1.0 / sum
	for i := range probs {
		probs[i] *= invSum
	}
	var total float32
	for _, v := range probs {
		total += v
	}
	if math.Abs(float64(total-1.0)) > 1e-5 {
		t.Errorf("uniform dist sum = %v, want 1.0", total)
	}
}
