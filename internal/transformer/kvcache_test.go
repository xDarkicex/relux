package transformer_test

import (
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestKVCache_NewEmpty(t *testing.T) {
	c := transformer.NewKVCache(4, transformer.Float32)
	if c.TotalLen(0) != 0 {
		t.Errorf("TotalLen(0) = %d, want 0 (no Append yet)", c.TotalLen(0))
	}
}

func TestKVCache_AppendFirst(t *testing.T) {
	c := transformer.NewKVCache(2, transformer.Float32)
	// [batch=1, heads=2, seq=1, headDim=3] -> 6 elements
	k := transformer.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 1, 3)
	v := transformer.NewTensor([]float32{7, 8, 9, 10, 11, 12}, 1, 2, 1, 3)
	c.Append(0, k, v)

	if c.TotalLen(0) != 1 {
		t.Errorf("TotalLen(0) = %d, want 1", c.TotalLen(0))
	}
	ck, cv := c.View(0)
	if ck == nil || cv == nil {
		t.Fatal("View returned nil")
	}
	if ck.Rank() != 4 || ck.Shape()[0] != 1 || ck.Shape()[1] != 2 || ck.Shape()[2] != 1 || ck.Shape()[3] != 3 {
		t.Errorf("K shape = %v, want [1 2 1 3]", ck.Shape())
	}
	// Verify the data was copied.
	wantK := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range ck.DataF32() {
		if v != wantK[i] {
			t.Errorf("K[%d] = %v, want %v", i, v, wantK[i])
		}
	}
}

func TestKVCache_AppendGrows(t *testing.T) {
	c := transformer.NewKVCache(1, transformer.Float32)
	// Two appends: first 1 token, then 2 more.
	k1 := transformer.NewTensor([]float32{1, 2, 3, 4}, 1, 1, 1, 4)
	v1 := transformer.NewTensor([]float32{5, 6, 7, 8}, 1, 1, 1, 4)
	c.Append(0, k1, v1)
	if c.TotalLen(0) != 1 {
		t.Fatalf("after 1st append, TotalLen=%d, want 1", c.TotalLen(0))
	}

	k2 := transformer.NewTensor([]float32{9, 10, 11, 12, 13, 14, 15, 16}, 1, 1, 2, 4)
	v2 := transformer.NewTensor([]float32{17, 18, 19, 20, 21, 22, 23, 24}, 1, 1, 2, 4)
	c.Append(0, k2, v2)
	if c.TotalLen(0) != 3 {
		t.Fatalf("after 2nd append, TotalLen=%d, want 3", c.TotalLen(0))
	}

	ck, _ := c.View(0)
	want := []float32{1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16}
	for i, v := range ck.DataF32() {
		if v != want[i] {
			t.Errorf("K[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestKVCache_Reset(t *testing.T) {
	c := transformer.NewKVCache(1, transformer.Float32)
	c.Append(0,
		transformer.NewTensor([]float32{1, 2, 3, 4}, 1, 1, 1, 4),
		transformer.NewTensor([]float32{5, 6, 7, 8}, 1, 1, 1, 4),
	)
	if c.TotalLen(0) != 1 {
		t.Fatalf("setup: TotalLen=%d, want 1", c.TotalLen(0))
	}
	c.Reset()
	if c.TotalLen(0) != 0 {
		t.Errorf("after Reset, TotalLen=%d, want 0", c.TotalLen(0))
	}
	ck, _ := c.View(0)
	if ck != nil {
		t.Errorf("after Reset, View(0).K = %v, want nil", ck)
	}
}

func TestKVCache_BFloat16(t *testing.T) {
	c := transformer.NewKVCache(1, transformer.BFloat16)
	k := transformer.NewTensor([]float32{1, 2, 3, 4}, 1, 1, 1, 4)
	v := transformer.NewTensor([]float32{5, 6, 7, 8}, 1, 1, 1, 4)
	c.Append(0, k, v)
	if c.TotalLen(0) != 1 {
		t.Errorf("TotalLen = %d, want 1", c.TotalLen(0))
	}
	ck, _ := c.View(0)
	if ck.DType() != transformer.BFloat16 {
		t.Errorf("K dtype = %s, want bf16", ck.DType())
	}
	// Verify the bf16 encoding (round-trip via float32).
	for i, v := range ck.DataBF16() {
		got := transformer.F32FromBF16(v)
		want := float32(i + 1)
		if got < want-0.01 || got > want+0.01 {
			t.Errorf("bf16[%d] round-trip = %v, want %v", i, got, want)
		}
	}
}
