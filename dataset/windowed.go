package dataset

import "io"

// WindowedIterator packs a contiguous []int token stream into
// input/target pairs using a sliding window.
//
// For each position i in the stream, Input = tokens[i:i+seqLen] and
// Target = tokens[i+1:i+seqLen+1]. stride controls the step between
// consecutive windows (stride=1 gives max density; stride=seqLen
// gives non-overlapping windows).
type WindowedIterator struct {
	tokens    []int
	seqLen    int
	batchSize int
	stride    int
	pos       int // current window start position
}

// NewWindowedIterator creates an iterator over an in-memory token
// stream. The stream should be a single contiguous sequence (e.g., a
// file's tokens concatenated with BOS/EOS separators).
//
// seqLen is the context window size. batchSize controls how many
// sequences are packed per batch. stride controls overlap between
// consecutive windows.
func NewWindowedIterator(tokens []int, seqLen int, batchSize int, stride int) *WindowedIterator {
	if batchSize <= 0 {
		batchSize = 1
	}
	if stride <= 0 {
		stride = 1
	}
	return &WindowedIterator{
		tokens:    tokens,
		seqLen:    seqLen,
		batchSize: batchSize,
		stride:    stride,
	}
}

// Next returns the next batch, or io.EOF when fewer than batchSize
// windows remain.
func (w *WindowedIterator) Next() (Batch, error) {
	windowSize := w.seqLen + 1
	if w.pos+windowSize > len(w.tokens) {
		return Batch{}, io.EOF
	}
	// Determine how many windows we can extract.
	remaining := (len(w.tokens) - w.pos - windowSize) / w.stride + 1
	nWindows := remaining
	if nWindows > w.batchSize {
		nWindows = w.batchSize
	}

	input := make([][]int, nWindows)
	target := make([][]int, nWindows)
	for i := 0; i < nWindows; i++ {
		start := w.pos + i*w.stride
		input[i] = w.tokens[start : start+w.seqLen]
		target[i] = w.tokens[start+1 : start+windowSize]
	}
	w.pos += nWindows * w.stride
	return Batch{Input: input, Target: target}, nil
}

// Reset rewinds the iterator to the beginning of the token stream.
func (w *WindowedIterator) Reset() {
	w.pos = 0
}
