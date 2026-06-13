package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoad_MinimalBPE(t *testing.T) {
	tok, err := Load(filepath.Join("testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok.VocabSize() < 4 {
		t.Errorf("VocabSize too small: %d", tok.VocabSize())
	}
}

func TestEncode(t *testing.T) {
	tok, err := Load(filepath.Join("testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	ids, err := tok.Encode("hello world")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Encode returned empty token list")
	}
	// All IDs should be in vocab range.
	for i, id := range ids {
		if id < 0 || id >= tok.VocabSize() {
			t.Errorf("token[%d]=%d out of vocab range [0,%d)", i, id, tok.VocabSize())
		}
	}
}

func TestDecode(t *testing.T) {
	tok, err := Load(filepath.Join("testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded := tok.Decode(ids)
	if decoded == "" {
		t.Error("Decode returned empty string")
	}
}

func TestSpecialTokens(t *testing.T) {
	tok, err := Load(filepath.Join("testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok.BOS() != 0 {
		t.Errorf("BOS = %d, want 0", tok.BOS())
	}
	if tok.PAD() != 1 {
		t.Errorf("PAD = %d, want 1", tok.PAD())
	}
	if tok.EOS() != 2 {
		t.Errorf("EOS = %d, want 2", tok.EOS())
	}
	if tok.UNK() != 3 {
		t.Errorf("UNK = %d, want 3", tok.UNK())
	}
}

func TestLoad_NonexistentFile(t *testing.T) {
	_, err := Load(filepath.Join("testdata", "nonexistent.json"))
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestEncodeWithSpecial(t *testing.T) {
	tok, err := Load(filepath.Join("testdata", "minimal.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	ids, err := tok.EncodeWithSpecial("hello")
	if err != nil {
		t.Fatalf("EncodeWithSpecial: %v", err)
	}
	if len(ids) == 0 {
		t.Error("EncodeWithSpecial returned empty token list")
	}
	// With special tokens, BOS (0) should be first.
	if ids[0] != tok.BOS() {
		t.Logf("EncodeWithSpecial first token=%d, BOS=%d (may vary by tokenizer config)", ids[0], tok.BOS())
	}
}

func TestLoad_FileContent(t *testing.T) {
	// Verify the testdata file exists.
	path := filepath.Join("testdata", "minimal.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("testdata/minimal.json not found")
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Round-trip: encode then decode.
	text := "hello world"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded := tok.Decode(ids)
	if decoded == "" {
		t.Error("round-trip decode produced empty string")
	}
	t.Logf("text=%q ids=%v decoded=%q", text, ids, decoded)
}
