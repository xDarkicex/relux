// Package dataset provides streaming data loading primitives.
package dataset

import (
	"io"
	"math/rand"
)

// ShuffledIterator wraps an Iterator and shuffles batches on each
// Reset. The first call to Reset (or the initial state) consumes
// all batches from the inner iterator into a buffer, shuffles them,
// and replays in shuffled order.
//
// Memory: the entire dataset is buffered in memory. For TB-scale
// corpora, prefer file-level shuffle (randomize file order in
// TextFileIterator) or use WindowedIterator with pre-shuffled token
// lists.
type ShuffledIterator struct {
	inner   Iterator
	rng     *rand.Rand
	batches []Batch
	pos     int
	loaded  bool
}

// NewShuffledIterator creates a ShuffledIterator wrapping inner.
// Shuffling uses rng; if rng is nil, rand.New(rand.NewSource(1))
// is used.
func NewShuffledIterator(inner Iterator, rng *rand.Rand) *ShuffledIterator {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	return &ShuffledIterator{inner: inner, rng: rng}
}

// Next returns the next batch in shuffled order. Returns io.EOF
// when all batches have been yielded.
func (s *ShuffledIterator) Next() (Batch, error) {
	if !s.loaded {
		s.load()
	}
	if s.pos >= len(s.batches) {
		return Batch{}, io.EOF
	}
	b := s.batches[s.pos]
	s.pos++
	return b, nil
}

// Reset re-shuffles batches for the next epoch. The inner iterator
// is reset and all batches are re-collected and shuffled.
func (s *ShuffledIterator) Reset() {
	s.inner.Reset()
	s.pos = 0
	s.loaded = false
	s.batches = s.batches[:0]
}

func (s *ShuffledIterator) load() {
	for {
		batch, err := s.inner.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		if len(batch.Input) == 0 {
			continue
		}
		// Deep-copy the batch so we own the data after inner
		// is reset.
		cp := Batch{
			Input:  make([][]int, len(batch.Input)),
			Target: make([][]int, len(batch.Target)),
		}
		for i := range batch.Input {
			cp.Input[i] = make([]int, len(batch.Input[i]))
			copy(cp.Input[i], batch.Input[i])
			cp.Target[i] = make([]int, len(batch.Target[i]))
			copy(cp.Target[i], batch.Target[i])
		}
		s.batches = append(s.batches, cp)
	}
	s.rng.Shuffle(len(s.batches), func(i, j int) {
		s.batches[i], s.batches[j] = s.batches[j], s.batches[i]
	})
	s.loaded = true
}
