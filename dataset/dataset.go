// Package dataset provides streaming data loading primitives for
// autoregressive language model training. It decouples tokenization
// from training so the relux Transformer can ingest corpora larger
// than available RAM.
//
// Core types:
//
//	Batch         — one batch of input/target token-ID slices
//	Iterator      — yields batches sequentially; resets for multi-epoch
//
// Concrete iterators:
//
//	WindowedIterator   — sliding windows over in-memory []int
//	TextFileIterator   — stream text files, tokenize on-the-fly
//	MmapIterator       — memory-map pre-tokenized binary files (zero-copy)
package dataset

import (
	"errors"
	"io"
)

// Batch is a training batch of input/target pairs for autoregressive
// language modeling. Input[t][s] is token s of sequence t, and
// Target[t][s] = Input[t][s+1] (one-step-ahead prediction).
type Batch struct {
	Input  [][]int // [batchSize][seqLen]
	Target [][]int // [batchSize][seqLen]
}

// Iterator yields batches sequentially from a data source. Next
// returns io.EOF when the source is exhausted. Reset reinitializes
// the source so Next can be called again (for multi-epoch training).
type Iterator interface {
	Next() (Batch, error)
	Reset()
}

// ErrEmptySource is returned by constructors when the token source
// produces fewer tokens than a single sequence requires.
var ErrEmptySource = errors.New("dataset: token source too short for one sequence")

// tokenBuffer is a simple ring buffer of ints used internally by
// streaming iterators to accumulate tokens and extract windows.
type tokenBuffer struct {
	buf    []int
	head   int // next write position (circular)
	length int // number of valid elements
}

func newTokenBuffer(cap int) *tokenBuffer {
	return &tokenBuffer{buf: make([]int, cap)}
}

func (b *tokenBuffer) push(tokens []int) {
	for _, t := range tokens {
		b.buf[b.head] = t
		b.head = (b.head + 1) % len(b.buf)
		if b.length < len(b.buf) {
			b.length++
		}
	}
}

func (b *tokenBuffer) len() int { return b.length }

// take extracts up to n tokens from the buffer (FIFO order).
// Used primarily in tests.
func (b *tokenBuffer) take(n int) []int {
	if n > b.length {
		n = b.length
	}
	if n <= 0 {
		return nil
	}
	out := make([]int, n)
	tail := (b.head - b.length + len(b.buf)) % len(b.buf)
	for i := 0; i < n; i++ {
		out[i] = b.buf[(tail+i)%len(b.buf)]
	}
	b.length -= n
	return out
}

// extractWindows pulls as many (seqLen+1)-sized windows as possible
// from the buffer and returns them as a single flat []int of
// concatenated windows. Each window is input+target: seqLen+1 tokens
// where input[:seqLen] and target[1:] form the pair.
//
// stride controls overlap. stride=1 means max density (each token
// position starts a new window); stride=seqLen means non-overlapping.
func (b *tokenBuffer) extractWindows(seqLen int, stride int, batchSize int) []int {
	windowSize := seqLen + 1
	if b.length < windowSize {
		return nil
	}
	// How many windows can we extract before running out?
	// After extracting k windows spaced by stride, we need:
	//   (k-1)*stride + windowSize <= b.length
	maxWindows := (b.length-windowSize)/stride + 1
	// Cap at batchSize.
	if maxWindows > batchSize {
		maxWindows = batchSize
	}
	if maxWindows <= 0 {
		return nil
	}

	tail := (b.head - b.length + len(b.buf)) % len(b.buf)
	out := make([]int, maxWindows*windowSize)
	for w := 0; w < maxWindows; w++ {
		start := (tail + w*stride) % len(b.buf)
		for j := 0; j < windowSize; j++ {
			out[w*windowSize+j] = b.buf[(start+j)%len(b.buf)]
		}
	}
	// Discard consumed tokens through the end of the last window.
	consumed := (maxWindows-1)*stride + windowSize
	b.length -= consumed
	return out
}

// packWindows converts a flat slice of concatenated windows into a Batch.
// Each window is windowSize tokens: Input[0:seqLen], Target[1:seqLen+1].
func packWindows(flat []int, seqLen int) Batch {
	windowSize := seqLen + 1
	nWindows := len(flat) / windowSize
	input := make([][]int, nWindows)
	target := make([][]int, nWindows)
	for w := 0; w < nWindows; w++ {
		base := w * windowSize
		input[w] = flat[base : base+seqLen]
		target[w] = flat[base+1 : base+windowSize]
	}
	return Batch{Input: input, Target: target}
}

// compile-time check
var _ Iterator = (*WindowedIterator)(nil)
var _ Iterator = (*TextFileIterator)(nil)
var _ Iterator = (*MmapIterator)(nil)

// ensureEOF wraps io.EOF so it can be compared with ==.
func isEOF(err error) bool { return err == io.EOF }
