// End-to-end load test against a real TinyLlama-1.1B safetensors file.
// Set RELUX_TEST_TINYLLAMA=/path/to/tinyllama.safetensors to run; otherwise
// skipped. This is the only relux test that needs a 2.2GB download.
package relux_test

import (
	"math/rand"
	"os"
	"testing"

	"github.com/xDarkicex/relux"
	"github.com/xDarkicex/relux/internal/transformer"
)

func topKIndices(s []float32, k int) []int {
	idx := make([]int, len(s))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < k && i < len(idx); i++ {
		maxJ := i
		for j := i + 1; j < len(idx); j++ {
			if s[idx[j]] > s[idx[maxJ]] {
				maxJ = j
			}
		}
		idx[i], idx[maxJ] = idx[maxJ], idx[i]
	}
	return idx[:k]
}

func TestLoadModel_TinyLlama_1B(t *testing.T) {
	path := os.Getenv("RELUX_TEST_TINYLLAMA")
	if path == "" {
		t.Skip("RELUX_TEST_TINYLLAMA not set; skipping real-model load test")
	}

	loaded, err := relux.LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if loaded.Architecture != "llama" {
		t.Errorf("architecture: want llama, got %q", loaded.Architecture)
	}
	if loaded.Tokenizer == nil {
		t.Error("tokenizer not loaded (tokenizer.json not found next to model)")
	}
	if loaded.Transformer == nil {
		t.Fatal("transformer is nil")
	}

	cfg := loaded.Transformer.Config()
	t.Logf("loaded: vocab=%d dModel=%d heads=%d kvHeads=%d layers=%d dFF=%d",
		cfg.VocabSize, cfg.DModel, cfg.NumHeads, cfg.NumKVHeads, cfg.NumLayers, cfg.DFF)

	if cfg.VocabSize != 32000 {
		t.Errorf("vocab: want 32000, got %d", cfg.VocabSize)
	}
	if cfg.DModel != 2048 {
		t.Errorf("dModel: want 2048, got %d", cfg.DModel)
	}
	if cfg.NumLayers != 22 {
		t.Errorf("layers: want 22, got %d", cfg.NumLayers)
	}
	if cfg.NumHeads != 32 || cfg.NumKVHeads != 4 {
		t.Errorf("heads: want 32/4, got %d/%d", cfg.NumHeads, cfg.NumKVHeads)
	}

	// Verify the weights were actually written into the correct slots
	// by reading back bf16 values and comparing to the source file.
	verifyWeightsWritten(t, loaded)

	// Generation: a loaded model should produce readable text.
	rng := rand.New(rand.NewSource(1))
	tokens, err := loaded.Transformer.Generate([]int{1}, 4, 0.7, 40, rng)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	t.Logf("generated: %v", tokens)
	if loaded.Tokenizer != nil {
		t.Logf("decoded: %q", loaded.Tokenizer.Decode(tokens))
	}

	// Self-prediction loss: for a properly-loaded model, asking it
	// to predict its own top-1 should give low loss.
	loaded.Transformer.SetMode(transformer.Inference)
	probeIn := []int{1, 2, 3, 4, 5, 6, 7, 8}
	probeOut := loaded.Transformer.Forward(probeIn, 1)
	probeData, _ := probeOut.ToF32()
	target := make([]int, 8)
	for i := 0; i < 8; i++ {
		target[i] = topKIndices(probeData[i*32000:(i+1)*32000], 1)[0]
	}
	loss, err := loaded.Transformer.TrainStep(probeIn, target, 1)
	if err != nil {
		t.Fatalf("TrainStep: %v", err)
	}
	t.Logf("self-prediction loss: %f (per-token: %f)", loss, loss/8)
	if loss > 30 {
		t.Errorf("self-prediction loss too high: %f — weights may not be loaded correctly", loss)
	}
}

// verifyWeightsWritten reads selected bf16 values from the loaded
// transformer and confirms they match the expected values from the
// source file.
func verifyWeightsWritten(t *testing.T, loaded *relux.LoadedModel) {
	t.Helper()
	// Read expected values from the source file.
	expected := map[string][4]uint16{
		"embed":   {14035, 13970, 13969, 14114},
		"wq0":     {47811, 47904, 48115, 48230},
		"wgate":   {15355, 15244, 15683, 15311},
		"normFin": {16374, 16361, 16377, 16382},
		"lmhead":  {15437, 15394, 15296, 48141},
	}
	// Read first 4 bf16s of the param at the expected slot.
	check := func(name string, slot int) {
		p := loaded.Transformer.Params()[slot]
		var got [4]uint16
		for i := 0; i < 4; i++ {
			got[i] = p.Data[i]
		}
		if got != expected[name] {
			t.Errorf("%s slot=%d: got %v, want %v", name, slot, got, expected[name])
		}
	}
	// Layout: [embed, (normAttn, Wq, Wk, Wv, Wo, normMlp, W1, W2, WGate)*L, finalNorm, lmHead.W, lmHead.b]
	check("embed", 0)
	check("wq0", 2)
	check("wgate", 9)
	check("normFin", 199)
	check("lmhead", 200)
}
