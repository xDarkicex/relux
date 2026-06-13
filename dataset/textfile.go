package dataset

import (
	"io"
	"os"

	"github.com/xDarkicex/relux/tokenizer"
)

// TextFileIterator streams text files, tokenizes them on-the-fly, and
// yields batches of windowed input/target pairs. It is suitable for
// corpora where individual files are small enough to tokenize in
// memory (e.g., .go source files, markdown documents).
//
// Files are read one at a time. Each file is fully tokenized, its
// token stream is appended to an internal ring buffer, and batches
// are extracted from the buffer as it fills.
type TextFileIterator struct {
	paths     []string
	tok       *tokenizer.Tokenizer
	seqLen    int
	batchSize int
	stride    int

	buf     *tokenBuffer
	fileIdx int // next file to read
	exhausted bool
}

// NewTextFileIterator creates a streaming iterator over text files.
// seqLen is the context window size; batchSize controls sequences per
// batch. The iterator maintains an internal ring buffer sized to hold
// batchSize * (seqLen+1) tokens.
func NewTextFileIterator(paths []string, tok *tokenizer.Tokenizer, seqLen int, batchSize int) *TextFileIterator {
	if batchSize <= 0 {
		batchSize = 1
	}
	// Size the buffer to hold at least batchSize windows plus some slack.
	bufCap := batchSize * (seqLen + 1) * 4
	if bufCap < 65536 {
		bufCap = 65536
	}
	return &TextFileIterator{
		paths:     paths,
		tok:       tok,
		seqLen:    seqLen,
		batchSize: batchSize,
		stride:    1,
		buf:       newTokenBuffer(bufCap),
	}
}

// Next returns the next batch. When the buffer runs low it reads and
// tokenizes the next file. Returns io.EOF when all files have been
// consumed and no more windows can be formed.
func (it *TextFileIterator) Next() (Batch, error) {
	windowSize := it.seqLen + 1
	for {
		// Try to extract a batch from the buffer.
		if it.buf.len() >= windowSize {
			flat := it.buf.extractWindows(it.seqLen, it.stride, it.batchSize)
			if len(flat) > 0 {
				return packWindows(flat, it.seqLen), nil
			}
		}
		// Buffer too low — read next file.
		if it.fileIdx >= len(it.paths) {
			if it.exhausted {
				return Batch{}, io.EOF
			}
			it.exhausted = true
			return Batch{}, io.EOF
		}
		if err := it.readNextFile(); err != nil {
			return Batch{}, err
		}
	}
}

// readNextFile reads the next file, tokenizes it, and appends the
// tokens to the ring buffer.
func (it *TextFileIterator) readNextFile() error {
	path := it.paths[it.fileIdx]
	it.fileIdx++
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	ids, err := it.tok.Encode(string(data))
	if err != nil {
		return err
	}
	if len(ids) > 0 {
		it.buf.push(ids)
	}
	return nil
}

// Reset rewinds to the beginning of the file list and clears the
// token buffer.
func (it *TextFileIterator) Reset() {
	it.fileIdx = 0
	it.exhausted = false
	it.buf = newTokenBuffer(len(it.buf.buf)) // same capacity, fresh buffer
}
