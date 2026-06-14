package dataset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/relux/tokenizer"
)

func newTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.Load(filepath.Join("..", "tokenizer", "testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("load test tokenizer: %v", err)
	}
	return tok
}

// --- tokenBuffer tests ---

func TestTokenBuffer_PushAndTake(t *testing.T) {
	buf := newTokenBuffer(10)
	buf.push([]int{1, 2, 3, 4, 5})
	if buf.len() != 5 {
		t.Fatalf("len = %d, want 5", buf.len())
	}
	out := buf.take(3)
	if len(out) != 3 || out[0] != 1 || out[1] != 2 || out[2] != 3 {
		t.Fatalf("take(3) = %v, want [1 2 3]", out)
	}
	if buf.len() != 2 {
		t.Fatalf("len after take = %d, want 2", buf.len())
	}
}

func TestTokenBuffer_WrapAround(t *testing.T) {
	buf := newTokenBuffer(5)
	// Fill buffer: [1,2,3,4,5]
	buf.push([]int{1, 2, 3, 4, 5})
	if buf.len() != 5 {
		t.Fatalf("len = %d, want 5", buf.len())
	}
	// Take 3: [1,2,3] removed, leaves [4,5]
	buf.take(3)
	// Push 2 more, wraps around: buf = [_, _, _, 4, 5], push 6 at 0, 7 at 1
	buf.push([]int{6, 7})
	if buf.len() != 4 {
		t.Fatalf("len = %d, want 4", buf.len())
	}
	// Should be [4,5,6,7]
	out := buf.take(4)
	if len(out) != 4 || out[0] != 4 || out[1] != 5 || out[2] != 6 || out[3] != 7 {
		t.Fatalf("take(4) = %v, want [4 5 6 7]", out)
	}
}

func TestTokenBuffer_ExtractWindows(t *testing.T) {
	buf := newTokenBuffer(100)
	// Push 15 tokens: 0..14
	tokens := make([]int, 15)
	for i := range tokens {
		tokens[i] = i
	}
	buf.push(tokens)
	// seqLen=3, stride=1, batchSize=3
	// Each window is 4 tokens (input+target).
	// Expected windows:
	//   [0,1,2,3] -> input [0,1,2], target [1,2,3]
	//   [1,2,3,4] -> input [1,2,3], target [2,3,4]
	//   [2,3,4,5] -> input [2,3,4], target [3,4,5]
	flat := buf.extractWindows(3, 1, 3)
	if len(flat) != 12 { // 3 windows * 4 tokens
		t.Fatalf("extractWindows returned %d tokens, want 12", len(flat))
	}
	expected := []int{0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5}
	for i, v := range expected {
		if flat[i] != v {
			t.Errorf("flat[%d] = %d, want %d", i, flat[i], v)
		}
	}
	// After extraction, buffer should have discarded (maxWindows-1)*stride + windowSize
	// = (3-1)*1 + 4 = 6 tokens, leaving 15-6=9 tokens (tokens 6..14)
	if buf.len() != 9 {
		t.Errorf("remaining len = %d, want 9", buf.len())
	}
}

// --- WindowedIterator tests ---

func TestWindowedIterator_Basic(t *testing.T) {
	// 10 tokens: 0..9
	tokens := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	it := NewWindowedIterator(tokens, 3, 2, 1)
	// seqLen=3, batchSize=2, stride=1
	// Batch 1: [0,1,2,3] and [1,2,3,4]
	batch, err := it.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if len(batch.Input) != 2 {
		t.Fatalf("batch size = %d, want 2", len(batch.Input))
	}
	// Window 0: input=[0,1,2], target=[1,2,3]
	for j := 0; j < 3; j++ {
		if batch.Input[0][j] != j {
			t.Errorf("input[0][%d] = %d, want %d", j, batch.Input[0][j], j)
		}
		if batch.Target[0][j] != j+1 {
			t.Errorf("target[0][%d] = %d, want %d", j, batch.Target[0][j], j+1)
		}
	}
}

func TestWindowedIterator_Exhaustion(t *testing.T) {
	tokens := []int{0, 1, 2, 3} // only 4 tokens
	it := NewWindowedIterator(tokens, 3, 2, 1)
	// seqLen=3 -> windowSize=4. Only 1 window fits.
	batch, err := it.Next()
	if err != nil {
		t.Fatalf("first Next: %v", err)
	}
	if len(batch.Input) != 1 {
		t.Fatalf("first batch size = %d, want 1", len(batch.Input))
	}
	_, err = it.Next()
	if !isEOF(err) {
		t.Errorf("second Next error = %v, want io.EOF", err)
	}
}

func TestWindowedIterator_Reset(t *testing.T) {
	tokens := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	it := NewWindowedIterator(tokens, 3, 4, 1)
	// First epoch.
	batch1, _ := it.Next()
	// Drain remaining.
	for {
		_, err := it.Next()
		if isEOF(err) {
			break
		}
	}
	// Reset and get first batch again.
	it.Reset()
	batch2, _ := it.Next()
	if len(batch1.Input) != len(batch2.Input) {
		t.Errorf("after reset batch size changed: %d vs %d", len(batch1.Input), len(batch2.Input))
	}
	if batch1.Input[0][0] != batch2.Input[0][0] {
		t.Errorf("after reset first token changed: %d vs %d", batch1.Input[0][0], batch2.Input[0][0])
	}
}

func TestWindowedIterator_Stride(t *testing.T) {
	tokens := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	// stride=2, seqLen=2 -> windowSize=3
	// Windows start at: 0, 2, 4, 6, 7 (7+3=10 so last window starts at 7)
	it := NewWindowedIterator(tokens, 2, 100, 2)
	batch, _ := it.Next()
	// Expected windows starting at 0, 2, 4, 6 = 4 windows
	if len(batch.Input) != 4 {
		t.Fatalf("stride=2 batch size = %d, want 4", len(batch.Input))
	}
	// First window: input=[0,1], target=[1,2]
	if batch.Input[0][0] != 0 || batch.Input[0][1] != 1 {
		t.Errorf("first input = %v, want [0 1]", batch.Input[0])
	}
	// Second window: input=[2,3], target=[3,4]
	if batch.Input[1][0] != 2 || batch.Input[1][1] != 3 {
		t.Errorf("second input = %v, want [2 3]", batch.Input[1])
	}
}

// --- MmapIterator tests ---

func TestPreprocess_RoundTrip(t *testing.T) {
	tok := newTestTokenizer(t)
	dir := t.TempDir()
	// Write two test files.
	file1 := filepath.Join(dir, "a.txt")
	file2 := filepath.Join(dir, "b.txt")
	os.WriteFile(file1, []byte("hello"), 0644)
	os.WriteFile(file2, []byte("world"), 0644)

	binPath := filepath.Join(dir, "tokens.bin")
	if err := PreprocessWithSeparators([]string{file1, file2}, tok, binPath); err != nil {
		t.Fatalf("PreprocessWithSeparators: %v", err)
	}

	it, err := NewMmapIterator(binPath, 3, 4)
	if err != nil {
		t.Fatalf("NewMmapIterator: %v", err)
	}
	batch, err := it.Next()
	if err != nil && !isEOF(err) {
		t.Fatalf("Next: %v", err)
	}
	if len(batch.Input) == 0 && !isEOF(err) {
		t.Error("expected non-empty batch or EOF")
	}
	if len(batch.Input) > 0 {
		for i, seq := range batch.Input {
			if len(seq) != 3 {
				t.Errorf("input[%d] len = %d, want 3", i, len(seq))
			}
		}
	}
}

// --- TextFileIterator tests ---

func TestTextFileIterator_Basic(t *testing.T) {
	tok := newTestTokenizer(t)
	dir := t.TempDir()
	file1 := filepath.Join(dir, "a.txt")
	file2 := filepath.Join(dir, "b.txt")
	// Write enough content to form at least one window.
	os.WriteFile(file1, []byte("hello world hello world hello world hello world"), 0644)
	os.WriteFile(file2, []byte("hello world hello world hello world hello world"), 0644)

	it := NewTextFileIterator([]string{file1, file2}, tok, 4, 2)
	batch, err := it.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if len(batch.Input) == 0 {
		t.Fatal("expected non-empty batch")
	}
	for i, seq := range batch.Input {
		if len(seq) != 4 {
			t.Errorf("input[%d] len = %d, want 4", i, len(seq))
		}
		if len(batch.Target[i]) != 4 {
			t.Errorf("target[%d] len = %d, want 4", i, len(batch.Target[i]))
		}
		// Verify target[t] == input[t+1] for overlapping positions.
		for j := 0; j < len(batch.Input[i])-1; j++ {
			if batch.Target[i][j] != batch.Input[i][j+1] {
				t.Errorf("target[%d][%d]=%d != input[%d][%d]=%d",
					i, j, batch.Target[i][j], i, j+1, batch.Input[i][j+1])
			}
		}
	}
}

func TestTextFileIterator_Reset(t *testing.T) {
	tok := newTestTokenizer(t)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello world hello world hello world"), 0644)

	it := NewTextFileIterator([]string{filepath.Join(dir, "a.txt")}, tok, 4, 2)
	batch1, err := it.Next()
	if err != nil {
		t.Fatalf("first Next: %v", err)
	}
	// Drain.
	for {
		_, err := it.Next()
		if isEOF(err) {
			break
		}
	}
	it.Reset()
	batch2, err := it.Next()
	if err != nil {
		t.Fatalf("after reset Next: %v", err)
	}
	if len(batch1.Input) == 0 || len(batch2.Input) == 0 {
		t.Fatal("empty batch after reset")
	}
	if batch1.Input[0][0] != batch2.Input[0][0] {
		t.Errorf("after reset first token changed: %d vs %d", batch1.Input[0][0], batch2.Input[0][0])
	}
}

func TestTextFileIterator_EmptyFile(t *testing.T) {
	tok := newTestTokenizer(t)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "empty.txt"), []byte{}, 0644)

	it := NewTextFileIterator([]string{filepath.Join(dir, "empty.txt")}, tok, 4, 2)
	_, err := it.Next()
	if !isEOF(err) {
		t.Errorf("expected EOF for empty file, got %v", err)
	}
}

func TestShuffledIterator_SameCount(t *testing.T) {
	tokens := make([]int, 200)
	for i := range tokens {
		tokens[i] = i % 10
	}
	inner := NewWindowedIterator(tokens, 8, 2, 1)
	si := NewShuffledIterator(inner, nil)

	// Count batches across two epochs (should be same).
	var count1, count2 int
	for {
		_, err := si.Next()
		if isEOF(err) {
			break
		}
		count1++
	}
	si.Reset()
	for {
		_, err := si.Next()
		if isEOF(err) {
			break
		}
		count2++
	}
	if count1 != count2 {
		t.Errorf("count1=%d, count2=%d after reset", count1, count2)
	}
	if count1 == 0 {
		t.Error("expected > 0 batches")
	}
}

func TestPrefetchIterator_BatchesMatch(t *testing.T) {
	tokens := make([]int, 200)
	for i := range tokens {
		tokens[i] = i % 10
	}
	inner := NewWindowedIterator(tokens, 8, 2, 1)
	pi := NewPrefetchIterator(inner, 4)

	var batches []Batch
	for {
		b, err := pi.Next()
		if isEOF(err) {
			break
		}
		batches = append(batches, b)
	}
	if len(batches) == 0 {
		t.Error("expected > 0 batches from prefetch")
	}

	// Reset and verify we get batches again.
	pi.Reset()
	b, err := pi.Next()
	if err != nil {
		t.Fatalf("Next after reset: %v", err)
	}
	if len(b.Input) == 0 {
		t.Error("empty batch after prefetch reset")
	}
}
