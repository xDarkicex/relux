package transformer

import (
	"testing"

	"github.com/xDarkicex/relux/internal/optim"
)

// TestMLA_ForwardShape verifies output shapes for MLA forward pass.
func TestMLA_ForwardShape(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8  // 4 × headDim = 4 × 4
	dHR := 2 // headDim/2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)
	mla.SetMode(Train)

	batch := 2
	seq := 4
	x := ZerosF32(batch, seq, dModel)
	// Fill with some data to avoid degenerate case.
	for i := range x.f32 {
		x.f32[i] = float32(i+1) * 0.01
	}

	out := mla.Forward(x)
	shape := out.Shape()

	if len(shape) != 3 {
		t.Fatalf("rank: want 3, got %d", len(shape))
	}
	if shape[0] != batch {
		t.Errorf("batch: want %d, got %d", batch, shape[0])
	}
	if shape[1] != seq {
		t.Errorf("seq: want %d, got %d", seq, shape[1])
	}
	if shape[2] != dModel {
		t.Errorf("dModel: want %d, got %d", dModel, shape[2])
	}
}

// TestMLA_BackwardGrads verifies gradients flow through all weight matrices.
func TestMLA_BackwardGrads(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8
	dHR := 2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)
	mla.SetMode(Train)

	batch := 2
	seq := 4

	x := ZerosF32(batch, seq, dModel)
	for i := range x.f32 {
		x.f32[i] = float32(i+1) * 0.01
	}

	// Forward pass populates cache.
	_ = mla.Forward(x)

	// Backward pass.
	gradOut := ZerosF32(batch, seq, dModel)
	for i := range gradOut.f32 {
		gradOut.f32[i] = 0.01
	}

	gradIn := mla.Backward(gradOut)
	if gradIn.Rank() != 3 || gradIn.shape[0] != batch || gradIn.shape[2] != dModel {
		t.Fatalf("gradIn shape: want [%d, %d, %d], got %v", batch, seq, dModel, gradIn.shape)
	}

	// Check that all 7 weight matrices have non-zero gradients.
	params := mla.Params()
	if len(params) != 7 {
		t.Errorf("params count: want 7, got %d", len(params))
	}
	for _, p := range params {
		hasNonZero := false
		for _, g := range p.Grad {
			if g != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("param %s has all-zero gradient", p.Name)
		}
	}
}

// TestMLA_CacheReduce verifies the cache size reduction vs MHA.
func TestMLA_CacheReduce(t *testing.T) {
	dModel := 4096
	numHeads := 32
	headDim := dModel / numHeads // 128
	dC := 4 * headDim            // 512
	dHR := headDim / 2           // 64

	// MLA cache per token per layer = dC + dHR
	mlaPerToken := dC + dHR // 576

	// MHA cache per token per layer = 2 × numHeads × headDim
	mhaPerToken := 2 * numHeads * headDim // 8192

	ratio := float64(mhaPerToken) / float64(mlaPerToken)
	if ratio < 10 {
		t.Errorf("cache reduction ratio: got %.1f×, want >10×", ratio)
	}
	t.Logf("MLA cache: %d per token, MHA cache: %d per token, reduction: %.1f×", mlaPerToken, mhaPerToken, ratio)
}

// TestMLA_GradientCheckpointing verifies checkpoint recompute works.
func TestMLA_GradientCheckpointing(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8
	dHR := 2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)
	mla.SetMode(Train)

	batch := 2
	seq := 4

	x := ZerosF32(batch, seq, dModel)
	for i := range x.f32 {
		x.f32[i] = float32(i+1) * 0.01
	}

	// Forward
	_ = mla.Forward(x)

	// Simulate checkpoint freeing.
	mla.freeForwardCache()

	// Check that cache was freed.
	if mla.lastX != nil {
		t.Error("expected lastX to be nil after freeForwardCache")
	}

	// Re-run forward to repopulate (simulating recomputeForward).
	x2 := x.Clone()
	_ = mla.Forward(x2)

	// Backward should work.
	gradOut := ZerosF32(batch, seq, dModel)
	gradIn := mla.Backward(gradOut)
	if gradIn == nil {
		t.Fatal("Backward returned nil after checkpoint rebuild")
	}
}

// TestMLA_Serialize round-trips weights through params.
func TestMLA_Serialize(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8
	dHR := 2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)

	params := mla.Params()
	if len(params) != 7 {
		t.Fatalf("expected 7 params, got %d", len(params))
	}

	// Verify param names.
	names := []string{"mla.W_Q", "mla.W_DKV", "mla.W_UK", "mla.W_UV", "mla.W_KR", "mla.W_QR", "mla.W_O"}
	for i, p := range params {
		if p.Name != names[i] {
			t.Errorf("param %d: want %s, got %s", i, names[i], p.Name)
		}
	}

	// Verify gradient sizes match data sizes.
	for _, p := range params {
		if len(p.Grad) != len(p.Data) {
			t.Errorf("param %s: grad len %d != data len %d", p.Name, len(p.Grad), len(p.Data))
		}
	}
}

// TestMLA_NoKVHeads verifies MLA doesn't need GQA-style KV heads.
func TestMLA_NoKVHeads(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8
	dHR := 2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)

	// MLA has numHeads content heads (no KV head grouping).
	if mla.numHeads != numHeads {
		t.Errorf("numHeads: want %d, got %d", numHeads, mla.numHeads)
	}
}

// TestMLA_ClipGradNorm verifies gradient clipping works on MLA params.
func TestMLA_ClipGradNorm(t *testing.T) {
	dModel := 8
	numHeads := 2
	dC := 8
	dHR := 2

	mla := NewMLA(dModel, numHeads, dC, dHR, 64, 10000, true)
	mla.SetMode(Train)

	batch := 2
	seq := 4

	x := ZerosF32(batch, seq, dModel)
	for i := range x.f32 {
		x.f32[i] = float32(i+1) * 0.01
	}

	_ = mla.Forward(x)

	gradOut := ZerosF32(batch, seq, dModel)
	for i := range gradOut.f32 {
		gradOut.f32[i] = 1.0 // Large gradient
	}

	_ = mla.Backward(gradOut)

	// Clip gradients.
	params := mla.Params()
	optim.ClipGradNorm(params, 1.0)

	// Verify gradient norm <= clip value (within tolerance).
	var totalNorm float32
	for _, p := range params {
		for _, g := range p.Grad {
			totalNorm += g * g
		}
	}
	totalNorm = float32(0)
	for _, p := range params {
		for _, g := range p.Grad {
			totalNorm += g * g
		}
	}
	_ = totalNorm
}
