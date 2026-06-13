// Package tokenizer provides a Go wrapper around HuggingFace tokenizer.json
// files using the sugarme/tokenizer library. It supports BPE, WordPiece, and
// WordLevel models, giving the relux Transformer access to any open-source
// tokenizer vocabulary.
//
// Usage:
//
//	tok, err := tokenizer.Load("tokenizer.json")
//	ids, err := tok.Encode("func main() { ... }")
//	text := tok.Decode(ids)
//	cfg.VocabSize = tok.VocabSize()
package tokenizer

import (
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer wraps a HuggingFace-compatible tokenizer loaded from a
// tokenizer.json file.
type Tokenizer struct {
	inner     *tokenizer.Tokenizer
	vocabSize int
}

// Load loads a tokenizer from a HuggingFace tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	tk, err := pretrained.FromFile(path)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{
		inner:     tk,
		vocabSize: tk.GetVocabSize(true),
	}, nil
}

// Encode tokenizes text into token IDs. Special tokens (BOS, EOS) are
// not automatically added; use EncodeWithSpecial for that.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	enc, err := t.inner.EncodeSingle(text, false)
	if err != nil {
		return nil, err
	}
	return enc.Ids, nil
}

// EncodeWithSpecial tokenizes text and adds special tokens
// (e.g., BOS/EOS for RoBERTa-style models).
func (t *Tokenizer) EncodeWithSpecial(text string) ([]int, error) {
	enc, err := t.inner.EncodeSingle(text, true)
	if err != nil {
		return nil, err
	}
	return enc.Ids, nil
}

// Decode converts token IDs back to text, skipping special tokens.
func (t *Tokenizer) Decode(tokens []int) string {
	return t.inner.Decode(tokens, true)
}

// VocabSize returns the vocabulary size including added tokens.
func (t *Tokenizer) VocabSize() int { return t.vocabSize }

// BOS returns the beginning-of-sequence token ID. Tries common
// conventions: <s>, [CLS], <|startoftext|>. Returns -1 if none found.
func (t *Tokenizer) BOS() int {
	if id, ok := t.inner.TokenToId("<s>"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("[CLS]"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("<|startoftext|>"); ok {
		return id
	}
	return -1
}

// EOS returns the end-of-sequence token ID. Tries common conventions:
// </s>, [SEP], <|endoftext|>. Returns -1 if none found.
func (t *Tokenizer) EOS() int {
	if id, ok := t.inner.TokenToId("</s>"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("[SEP]"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("<|endoftext|>"); ok {
		return id
	}
	return -1
}

// PAD returns the padding token ID. Tries common conventions:
// <pad>, [PAD]. Returns -1 if none found.
func (t *Tokenizer) PAD() int {
	if id, ok := t.inner.TokenToId("<pad>"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("[PAD]"); ok {
		return id
	}
	return -1
}

// UNK returns the unknown token ID. Tries common conventions:
// <unk>, [UNK]. Returns -1 if none found.
func (t *Tokenizer) UNK() int {
	if id, ok := t.inner.TokenToId("<unk>"); ok {
		return id
	}
	if id, ok := t.inner.TokenToId("[UNK]"); ok {
		return id
	}
	return -1
}
