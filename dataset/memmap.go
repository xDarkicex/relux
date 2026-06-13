package dataset

import (
	"encoding/binary"
	"io"
	"os"

	"github.com/xDarkicex/relux/tokenizer"
)

// MmapIterator reads a pre-tokenized binary file and yields windowed
// batches. The binary file is a flat sequence of little-endian int32
// token IDs. This is the recommended path for production training:
// pre-tokenize the corpus once, then train many epochs without
// re-tokenizing.
//
// For files up to a few GB the entire file is read into memory; the
// OS page cache handles the rest. For larger files a ring-buffer
// streaming approach can be added later.
type MmapIterator struct {
	tokens    []int32
	seqLen    int
	batchSize int
	stride    int
	pos       int // current window start position
}

// NewMmapIterator opens a pre-tokenized binary file and returns an
// iterator over its content.
//
// Binary format: flat []int32, little-endian. Each int32 is a token
// ID. Files are typically produced by Preprocess().
func NewMmapIterator(path string, seqLen int, batchSize int) (*MmapIterator, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	if len(data)%4 != 0 {
		return nil, io.ErrUnexpectedEOF
	}
	tokens := make([]int32, len(data)/4)
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	return &MmapIterator{
		tokens:    tokens,
		seqLen:    seqLen,
		batchSize: batchSize,
		stride:    1,
	}, nil
}

// Next returns the next batch, or io.EOF when fewer than batchSize
// windows remain.
func (m *MmapIterator) Next() (Batch, error) {
	windowSize := m.seqLen + 1
	if m.pos+windowSize > len(m.tokens) {
		return Batch{}, io.EOF
	}
	remaining := (len(m.tokens) - m.pos - windowSize) / m.stride + 1
	nWindows := remaining
	if nWindows > m.batchSize {
		nWindows = m.batchSize
	}
	input := make([][]int, nWindows)
	target := make([][]int, nWindows)
	for i := 0; i < nWindows; i++ {
		start := m.pos + i*m.stride
		in := make([]int, m.seqLen)
		tg := make([]int, m.seqLen)
		for j := 0; j < m.seqLen; j++ {
			in[j] = int(m.tokens[start+j])
			tg[j] = int(m.tokens[start+1+j])
		}
		input[i] = in
		target[i] = tg
	}
	m.pos += nWindows * m.stride
	return Batch{Input: input, Target: target}, nil
}

// Reset rewinds to the beginning of the file.
func (m *MmapIterator) Reset() { m.pos = 0 }

// Preprocess tokenizes a list of text files and writes a single
// binary file of little-endian int32 token IDs. Each file's tokens
// are concatenated directly (no BOS/EOS separators — the caller
// should handle document boundaries by injecting special tokens
// before calling Preprocess, or use PreprocessWithSeparators).
//
// The output file is suitable for use with NewMmapIterator.
func Preprocess(paths []string, tok *tokenizer.Tokenizer, outputPath string) error {
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer out.Close()
	var buf [4]byte
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		ids, err := tok.Encode(string(data))
		if err != nil {
			return err
		}
		for _, id := range ids {
			binary.LittleEndian.PutUint32(buf[:], uint32(id))
			if _, err := out.Write(buf[:]); err != nil {
				return err
			}
		}
	}
	return nil
}

// PreprocessWithSeparators tokenizes text files and writes a binary
// file, inserting BOS and EOS tokens between documents.
func PreprocessWithSeparators(paths []string, tok *tokenizer.Tokenizer, outputPath string) error {
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer out.Close()
	var buf [4]byte
	bos := tok.BOS()
	eos := tok.EOS()
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		ids, err := tok.Encode(string(data))
		if err != nil {
			return err
		}
		if bos >= 0 {
			binary.LittleEndian.PutUint32(buf[:], uint32(bos))
			out.Write(buf[:])
		}
		for _, id := range ids {
			binary.LittleEndian.PutUint32(buf[:], uint32(id))
			out.Write(buf[:])
		}
		if eos >= 0 {
			binary.LittleEndian.PutUint32(buf[:], uint32(eos))
			out.Write(buf[:])
		}
	}
	return nil
}
