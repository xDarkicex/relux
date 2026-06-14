package relux

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/xDarkicex/relux/internal/optim"
	"github.com/xDarkicex/relux/internal/serialize"
	"github.com/xDarkicex/relux/internal/transformer"
)

// archEntryCount returns the number of arch entries in a
// Transformer with L blocks. The arch is a flat list:
//
//	[RoPE, Embedding, (RMSNorm, MHA, RMSNorm, MLP) * L,
//	 finalNorm, lmHead]
//
// Total = 2 + 4*L + 1 = 4*L + 3.
func archEntryCount(numBlocks int) int {
	return 4*numBlocks + 3
}

// totalParamCount returns the total number of f32 weights
// in the live Transformer. Used for the header's
// total_params field.
func totalParamCount(t *Transformer) int64 {
	var total int64
	for _, p := range t.Params() {
		total += int64(len(p.Data))
	}
	return total
}

// Save writes the Transformer to w in the v1 binary
// format. The optimizer state is included; pass nil to
// save a transformer with no optimizer state (the file
// will still have an empty state block — the reader
// returns nil state).
func (t *Transformer) Save(w io.Writer) error {
	return writeTransformerV1(w, t, t.getOptimizerState())
}

// writeTransformerV1 walks the live Transformer in
// canonical order and emits the v1 file. The order is:
//
//	header
//	arch: [RoPE, Embedding, (RMSNorm, MHA, RMSNorm, MLP) * L,
//	       finalNorm, lmHead]
//	weights: same order, with each layer's params in the
//	         order returned by Params()
//	optimizer state (Adam, all params)
//	footer (SHA-256 of the body)
func writeTransformerV1(w io.Writer, t *Transformer, state *optim.State) error {
	cfg := t.Config()
	headDim := cfg.DModel / cfg.NumHeads
	numEntries := archEntryCount(cfg.NumLayers)
	totalParams := totalParamCount(t)
	wr := serialize.NewV1Writer(w, numEntries, totalParams)
	if err := wr.WriteHeader(); err != nil {
		return err
	}
	// RoPE
	if err := wr.WriteArchEntry(serialize.LayerTagRoPE,
		[]uint32{uint32(headDim), uint32(cfg.MaxSeqLen)},
		[]float32{cfg.RopeBase}); err != nil {
		return err
	}
	// Embedding arch + weight
	if err := wr.WriteArchEntry(serialize.LayerTagEmbedding,
		[]uint32{uint32(cfg.VocabSize), uint32(cfg.DModel)},
		nil); err != nil {
		return err
	}
	if err := writeEmbeddingWeights(wr, t.GetEmbedding()); err != nil {
		return err
	}
	// Blocks
	for i, b := range t.GetBlocks() {
		// Pre-attn norm
		if err := wr.WriteArchEntry(serialize.LayerTagRMSNorm,
			[]uint32{uint32(cfg.DModel)},
			[]float32{b.NormAttn().Eps()}); err != nil {
			return fmt.Errorf("block %d normAttn arch: %w", i, err)
		}
		// MHA
		if err := wr.WriteArchEntry(serialize.LayerTagMHA,
			[]uint32{uint32(cfg.DModel), uint32(cfg.NumHeads), uint32(cfg.NumKVHeads), uint32(headDim)},
			nil); err != nil {
			return fmt.Errorf("block %d mha arch: %w", i, err)
		}
		// Pre-MLP norm
		if err := wr.WriteArchEntry(serialize.LayerTagRMSNorm,
			[]uint32{uint32(cfg.DModel)},
			[]float32{b.NormMlp().Eps()}); err != nil {
			return fmt.Errorf("block %d normMlp arch: %w", i, err)
		}
		// MLP
		if err := wr.WriteArchEntry(serialize.LayerTagMLP,
			[]uint32{uint32(cfg.DModel), uint32(cfg.DFF)},
			nil); err != nil {
			return fmt.Errorf("block %d mlp arch: %w", i, err)
		}
		// Block weights: normAttn.gamma, mha.Wq Wk Wv Wo,
		// normMlp.gamma, mlp.W1 b1 W2 b2
		if err := writeRMSNormWeights(wr, b.NormAttn()); err != nil {
			return fmt.Errorf("block %d normAttn weights: %w", i, err)
		}
		if err := writeMHAWeights(wr, b.BlockMHA()); err != nil {
			return fmt.Errorf("block %d mha weights: %w", i, err)
		}
		if err := writeRMSNormWeights(wr, b.NormMlp()); err != nil {
			return fmt.Errorf("block %d normMlp weights: %w", i, err)
		}
		if err := writeMLPWeights(wr, b.BlockMLP()); err != nil {
			return fmt.Errorf("block %d mlp weights: %w", i, err)
		}
	}
	// finalNorm
	if err := wr.WriteArchEntry(serialize.LayerTagRMSNorm,
		[]uint32{uint32(cfg.DModel)},
		[]float32{t.GetFinalNorm().Eps()}); err != nil {
		return err
	}
	if err := writeRMSNormWeights(wr, t.GetFinalNorm()); err != nil {
		return err
	}
	// lmHead
	if err := wr.WriteArchEntry(serialize.LayerTagLinear,
		[]uint32{uint32(cfg.DModel), uint32(cfg.VocabSize)},
		nil); err != nil {
		return err
	}
	if err := writeLinearWeights(wr, t.GetLMHead()); err != nil {
		return err
	}
	// Optimizer state. The state may be nil; write an
	// empty block.
	if state != nil {
		kind, step, adamStates := serialize.FromOptimState(state)
		if err := wr.WriteOptimizerState(kind, step, adamStates); err != nil {
			return err
		}
	} else {
		if err := wr.WriteOptimizerState("adam", 0, nil); err != nil {
			return err
		}
	}
	if _, err := wr.WriteFooter(); err != nil {
		return err
	}
	return nil
}

// LoadTransformer reads a v1 .relux file from r. Use
// (*Transformer).SetOptimizerState to install the
// returned optimizer state on a fresh Adam before
// resuming training.
//
// LoadTransformer only supports v1 files. For v0 (gob)
// *Network files, use (*Network).Load.
func LoadTransformer(r io.Reader) (*Transformer, *optim.State, error) {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReader(r)
	}
	magic, err := br.Peek(4)
	if err != nil {
		return nil, nil, fmt.Errorf("LoadTransformer: read magic: %w", err)
	}
	if !bytes.Equal(magic, []byte{'R', 'E', 'L', 'V'}) {
		return nil, nil, errors.New("LoadTransformer: file is not a v1 .relux (no RELV magic); use Network.Load for v0 gob files")
	}
	return readTransformerV1(br)
}

// readTransformerV1 reads the v1 file and reconstructs
// the Transformer. The reader walks the arch in canonical
// order; each entry is matched against the construction
// order in NewTransformer.
//
// Limitation: this v1 reader doesn't dynamically size the
// Transformer from the file's arch — it requires a
// caller-provided ConfigTransformer. The header's
// NumLayers and total_params are sanity-checked but the
// reader uses its own knowledge of the arch layout to
// interpret the file. This is fine for the v1 contract:
// the file is always written by Save, and Save always
// writes a Transformer built by NewTransformer. The
// reader gets the dims from the file (dModel, numHeads,
// numKVHeads, dFF, vocabSize, etc.) and reconstructs
// the same architecture.
//
// The only twist: we don't know numBlocks until we count
// the entries. We accept a 0-value Transformer (the
// caller built one) and then walk the file to verify
// the dims match. We then replace the layers' params in
// place.
func readTransformerV1(r io.Reader) (*Transformer, *optim.State, error) {
	rdr, err := serialize.NewV1Reader(r)
	if err != nil {
		return nil, nil, err
	}
	numLayers := int(rdr.Header().NumLayers)
	// Walk the arch in canonical order, collecting dims
	// and writing weights into a freshly-constructed
	// Transformer. We collect the dims as we read, then
	// construct the Transformer once we have them all.
	// But we need the Transformer to be live *during* the
	// read in order to install weights in place.
	//
	// Solution: read the RoPE dims to get headDim and
	// maxSeqLen. Read the Embedding dims to get vocabSize
	// and dModel. Read the first MHA to get numHeads and
	// numKVHeads. Read the first MLP to get dFF. Then
	// construct the Transformer. Then continue reading
	// and install the weights.
	//
	// Layer 0: RoPE
	tag, dims, floats, err := rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read RoPE arch: %w", err)
	}
	if tag != serialize.LayerTagRoPE {
		return nil, nil, fmt.Errorf("read RoPE: tag %d, want RoPE=%d", tag, serialize.LayerTagRoPE)
	}
	headDim := int(dims[0])
	maxSeqLen := int(dims[1])
	ropeBase := floats[0]
	// Layer 1: Embedding arch
	tag, dims, _, err = rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read Embedding arch: %w", err)
	}
	if tag != serialize.LayerTagEmbedding {
		return nil, nil, fmt.Errorf("read Embedding: tag %d, want Embedding=%d", tag, serialize.LayerTagEmbedding)
	}
	vocabSize := int(dims[0])
	dModel := int(dims[1])
	// Read the embedding weight.
	embedW, err := rdr.ReadWeight()
	if err != nil {
		return nil, nil, fmt.Errorf("read Embedding weight: %w", err)
	}
	// Sanity: 4*L+3 == numLayers, so L = (numLayers-3)/4
	if numLayers < 3 || (numLayers-3)%4 != 0 {
		return nil, nil, fmt.Errorf("v1: header numLayers=%d is not of the form 4*L+3", numLayers)
	}
	numBlocks := (numLayers - 3) / 4
	// We need numHeads, numKVHeads, dFF, normEps from the
	// first block. Read all four arch entries for block 0.
	// normAttn arch
	tag, _, floats, err = rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read block 0 normAttn arch: %w", err)
	}
	if tag != serialize.LayerTagRMSNorm {
		return nil, nil, fmt.Errorf("read block 0 normAttn: tag %d, want RMSNorm=%d", tag, serialize.LayerTagRMSNorm)
	}
	normEps := floats[0]
	// MHA arch
	tag, dims, _, err = rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read block 0 mha arch: %w", err)
	}
	if tag != serialize.LayerTagMHA {
		return nil, nil, fmt.Errorf("read block 0 mha: tag %d, want MHA=%d", tag, serialize.LayerTagMHA)
	}
	numHeads := int(dims[1])
	numKVHeads := int(dims[2])
	_ = dims[3] // headDim, must match
	// normMlp arch
	tag, _, _, err = rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read block 0 normMlp arch: %w", err)
	}
	if tag != serialize.LayerTagRMSNorm {
		return nil, nil, fmt.Errorf("read block 0 normMlp: tag %d, want RMSNorm=%d", tag, serialize.LayerTagRMSNorm)
	}
	// MLP arch
	tag, dims, _, err = rdr.ReadArchEntry()
	if err != nil {
		return nil, nil, fmt.Errorf("read block 0 mlp arch: %w", err)
	}
	if tag != serialize.LayerTagMLP {
		return nil, nil, fmt.Errorf("read block 0 mlp: tag %d, want MLP=%d", tag, serialize.LayerTagMLP)
	}
	dFF := int(dims[1])
	// Now we can construct the Transformer.
	cfg := ConfigTransformer{
		VocabSize:  vocabSize,
		DModel:     dModel,
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		NumLayers:  numBlocks,
		DFF:        dFF,
		MaxSeqLen:  maxSeqLen,
		RopeBase:   ropeBase,
		NormEps:    normEps,
		Causal:     true,
	}
	t, err := NewTransformer(cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("construct transformer: %w", err)
	}
	// Replace the rope with one built from the file's
	// (headDim, base, maxSeqLen) — same values, but
	// guaranteeing the cos/sin tables are computed with
	// the file's base.
	t.SetRoPE(transformer.NewRotaryEmbedding(headDim, ropeBase, maxSeqLen))
	// Install the embedding weight. The wire format is
	// bf16; the in-memory master is also bf16 (per the
	// optim.Param contract). Direct copy.
	t.GetEmbedding().GetParam().Data = embedW
	// Read block 0 weights: normAttn.gamma, mha.Wq Wk Wv Wo,
	// normMlp.gamma, mlp.W1 b1 W2 b2
	if err := readRMSNormWeights(rdr, t.GetBlocks()[0].NormAttn()); err != nil {
		return nil, nil, fmt.Errorf("read block 0 normAttn weight: %w", err)
	}
	if err := readMHAWeights(rdr, t.GetBlocks()[0].BlockMHA()); err != nil {
		return nil, nil, fmt.Errorf("read block 0 mha weights: %w", err)
	}
	if err := readRMSNormWeights(rdr, t.GetBlocks()[0].NormMlp()); err != nil {
		return nil, nil, fmt.Errorf("read block 0 normMlp weight: %w", err)
	}
	if err := readMLPWeights(rdr, t.GetBlocks()[0].BlockMLP()); err != nil {
		return nil, nil, fmt.Errorf("read block 0 mlp weights: %w", err)
	}
	// Read blocks 1..numBlocks-1 (arch + weights).
	for i := 1; i < numBlocks; i++ {
		// normAttn arch
		if _, _, _, err = rdr.ReadArchEntry(); err != nil {
			return nil, nil, fmt.Errorf("read block %d normAttn arch: %w", i, err)
		}
		// mha arch
		if _, _, _, err = rdr.ReadArchEntry(); err != nil {
			return nil, nil, fmt.Errorf("read block %d mha arch: %w", i, err)
		}
		// normMlp arch
		if _, _, _, err = rdr.ReadArchEntry(); err != nil {
			return nil, nil, fmt.Errorf("read block %d normMlp arch: %w", i, err)
		}
		// mlp arch
		if _, _, _, err = rdr.ReadArchEntry(); err != nil {
			return nil, nil, fmt.Errorf("read block %d mlp arch: %w", i, err)
		}
		// weights
		b := t.GetBlocks()[i]
		if err := readRMSNormWeights(rdr, b.NormAttn()); err != nil {
			return nil, nil, fmt.Errorf("read block %d normAttn weight: %w", i, err)
		}
		if err := readMHAWeights(rdr, b.BlockMHA()); err != nil {
			return nil, nil, fmt.Errorf("read block %d mha weights: %w", i, err)
		}
		if err := readRMSNormWeights(rdr, b.NormMlp()); err != nil {
			return nil, nil, fmt.Errorf("read block %d normMlp weight: %w", i, err)
		}
		if err := readMLPWeights(rdr, b.BlockMLP()); err != nil {
			return nil, nil, fmt.Errorf("read block %d mlp weights: %w", i, err)
		}
	}
	// finalNorm arch + weight
	if _, _, _, err = rdr.ReadArchEntry(); err != nil {
		return nil, nil, fmt.Errorf("read finalNorm arch: %w", err)
	}
	if err := readRMSNormWeights(rdr, t.GetFinalNorm()); err != nil {
		return nil, nil, fmt.Errorf("read finalNorm weight: %w", err)
	}
	// lmHead arch + weights
	if _, _, _, err = rdr.ReadArchEntry(); err != nil {
		return nil, nil, fmt.Errorf("read lmHead arch: %w", err)
	}
	if err := readLinearWeights(rdr, t.GetLMHead()); err != nil {
		return nil, nil, fmt.Errorf("read lmHead weights: %w", err)
	}
	// Optimizer state.
	kind, step, states, err := rdr.ReadOptimizerState()
	if err != nil {
		return nil, nil, fmt.Errorf("read optim state: %w", err)
	}
	var state *optim.State
	if kind != "" {
		state, err = serialize.ToOptimState(kind, step, states)
		if err != nil {
			return nil, nil, err
		}
	}
	if err := rdr.ReadFooter(); err != nil {
		return nil, nil, err
	}
	return t, state, nil
}

// writeEmbeddingWeights writes the embedding weight.
func writeEmbeddingWeights(wr *serialize.V1Writer, e *transformer.Embedding) error {
	return wr.WriteWeight(e.GetParam().Data)
}

// writeRMSNormWeights writes a gamma vector.
func writeRMSNormWeights(wr *serialize.V1Writer, rn *transformer.RMSNorm) error {
	return wr.WriteWeight(rn.GetParam().Data)
}

// writeMHAWeights writes Wq, Wk, Wv, Wo.
func writeMHAWeights(wr *serialize.V1Writer, m *transformer.MHA) error {
	if err := wr.WriteWeight(m.WqParam().Data); err != nil {
		return err
	}
	if err := wr.WriteWeight(m.WkParam().Data); err != nil {
		return err
	}
	if err := wr.WriteWeight(m.WvParam().Data); err != nil {
		return err
	}
	return wr.WriteWeight(m.WoParam().Data)
}

// writeMLPWeights writes W1, b1, W2, b2.
func writeMLPWeights(wr *serialize.V1Writer, m *transformer.MLP) error {
	if err := wr.WriteWeight(m.W1Param().Data); err != nil {
		return err
	}
	if err := wr.WriteWeight(m.B1Param().Data); err != nil {
		return err
	}
	if err := wr.WriteWeight(m.W2Param().Data); err != nil {
		return err
	}
	return wr.WriteWeight(m.B2Param().Data)
}

// writeLinearWeights writes W then b.
func writeLinearWeights(wr *serialize.V1Writer, l *transformer.Linear) error {
	if err := wr.WriteWeight(l.WParam().Data); err != nil {
		return err
	}
	return wr.WriteWeight(l.BParam().Data)
}

// readRMSNormWeights reads one gamma vector and installs
// it in place.
func readRMSNormWeights(rdr *serialize.V1Reader, rn *transformer.RMSNorm) error {
	g, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	rn.GetParam().Data = g
	return nil
}

// readMHAWeights reads Wq, Wk, Wv, Wo and installs each in
// place.
func readMHAWeights(rdr *serialize.V1Reader, m *transformer.MHA) error {
	wq, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	wk, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	wv, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	wo, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	m.WqParam().Data = wq
	m.WkParam().Data = wk
	m.WvParam().Data = wv
	m.WoParam().Data = wo
	return nil
}

// readMLPWeights reads W1, b1, W2, b2 and installs each in
// place.
func readMLPWeights(rdr *serialize.V1Reader, m *transformer.MLP) error {
	w1, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	b1, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	w2, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	b2, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	m.W1Param().Data = w1
	m.B1Param().Data = b1
	m.W2Param().Data = w2
	m.B2Param().Data = b2
	return nil
}

// readLinearWeights reads W then b and installs them.
func readLinearWeights(rdr *serialize.V1Reader, l *transformer.Linear) error {
	w, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	b, err := rdr.ReadWeight()
	if err != nil {
		return err
	}
	l.WParam().Data = w
	l.BParam().Data = b
	return nil
}

// SaveFile writes the Transformer to a file.
func (t *Transformer) SaveFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("SaveFile: create %s: %w", path, err)
	}
	defer f.Close()
	return t.Save(f)
}

// LoadTransformerFile reads a v1 .relux file.
func LoadTransformerFile(path string) (*Transformer, *optim.State, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("LoadTransformerFile: open %s: %w", path, err)
	}
	defer f.Close()
	return LoadTransformer(f)
}

// SetOptimizerState installs an optim.State on the
// transformer's owned Adam. Use this after LoadTransformer
// to resume training with the saved state.
func (t *Transformer) SetOptimizerState(state *optim.State) error {
	if state == nil {
		return nil
	}
	if t.adam == nil {
		t.adam = &optim.Adam{
			LR:    0,
			Beta1: 0.9,
			Beta2: 0.999,
			Eps:   1e-8,
		}
	}
	t.optimState = state
	return t.adam.LoadState(*state)
}

// getOptimizerState returns the current optim.State of the
// transformer's Adam. Returns the cached state if available,
// otherwise captures the live Adam state. Returns nil only
// if no Adam has ever been installed.
func (t *Transformer) getOptimizerState() *optim.State {
	if t.adam == nil {
		return nil
	}
	if t.optimState != nil {
		return t.optimState
	}
	st := t.adam.State()
	return &st
}
