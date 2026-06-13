package transformer

// GenerateToken steps the Transformer one token at a time
// in inference mode. This is the autoregressive loop used
// by Transformer.Generate (and exposed standalone for
// testing). The Transformer must be in Inference mode.
//
// The flow per step:
//   1. Embed the input token IDs.
//   2. Run each block (in Inference mode, MHA reads from
//      its KV-cache).
//   3. Apply the final RMSNorm and the lm_head (a Dense
//      mapping dModel -> vocabSize).
//   4. Sample the next token from the last position's logits.
//   5. Append the new K/V to each block's KV-cache.
//
// The KV-cache is maintained per-block; AppendKV is a
// per-block op that the caller (or the higher-level
// Generate) drives.
type GenerateState struct {
	sampler *Sampler
	eosID   int
	maxLen  int
}

// NewGenerateState returns a GenerateState for use with
// Transformer.Generate. eosID is the token ID that, if
// sampled, terminates generation. maxLen caps the total
// number of generated tokens.
func NewGenerateState(sampler *Sampler, eosID, maxLen int) *GenerateState {
	return &GenerateState{sampler: sampler, eosID: eosID, maxLen: maxLen}
}
