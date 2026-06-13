package transformer

import (
	"fmt"

	"github.com/xDarkicex/relux/internal/alloc"
)

// KVCache holds the per-layer K and V state for autoregressive
// generation. During decode, the new K/V (for the just-generated
// token) is appended to the cache; the next attention call
// reads the full sequence from K[0:totalLen] and V[0:totalLen].
//
// Storage is bfloat16 (the active dtype) when constructed with
// the bfloat16 path; the float32 path is used for training
// and for the bfloat16-roundtrip tests.
//
// Total memory is:
//   2 * numLayers * totalLen * numKVHeads * headDim * 2 bytes
// For 1M-token / 12-layer / 8-KV-head / 128-head-dim / bf16:
//   2 * 12 * 1e6 * 8 * 128 * 2 = 47 GB.
// The caller is expected to size maxLen to their hardware.
type KVCache struct {
	// Per-layer [K, V] tensors. Initialized lazily by
	// Append so the caller can pre-allocate zero-length
	// caches that grow as needed.
	layers []layerKV
	dtype  DType
}

type layerKV struct {
	K     *Tensor
	V     *Tensor
	total int // number of positions stored (rows along the seq dim)
}

// NewKVCache constructs an empty KV-cache for the given number
// of layers, with the active dtype. Each layer's K and V is
// nil until the first Append for that layer.
func NewKVCache(numLayers int, dtype DType) *KVCache {
	if numLayers <= 0 {
		panic(fmt.Sprintf("NewKVCache: numLayers=%d, must be > 0", numLayers))
	}
	return &KVCache{
		layers: make([]layerKV, numLayers),
		dtype:  dtype,
	}
}

// DType returns the active dtype of the cache.
func (c *KVCache) DType() DType { return c.dtype }

// Reset clears all stored K/V. Used between independent
// generation sequences.
func (c *KVCache) Reset() {
	for i := range c.layers {
		if c.layers[i].K != nil {
			alloc.Free(c.layers[i].K.f32)
			alloc.Free(c.layers[i].K.f64)
			alloc.Free(c.layers[i].K.bf16)
		}
		if c.layers[i].V != nil {
			alloc.Free(c.layers[i].V.f32)
			alloc.Free(c.layers[i].V.f64)
			alloc.Free(c.layers[i].V.bf16)
		}
		c.layers[i] = layerKV{}
	}
}

// Append concatenates newK and newV onto layerIdx's K and V.
// Both newK and newV must have the same shape; their seq dim
// (the second-to-last dim) is concatenated. The KV-cache's
// own seq dim becomes the existing length + newLen.
//
// newK and newV are NOT consumed; the caller still owns them
// and is responsible for freeing them (typically by re-using
// the active buffer pattern — Append copies into the cache's
// own storage).
func (c *KVCache) Append(layerIdx int, newK, newV *Tensor) {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		panic(fmt.Sprintf("KVCache.Append: layerIdx=%d out of range [0, %d)", layerIdx, len(c.layers)))
	}
	if !sameShape(newK.shape, newV.shape) {
		panic(fmt.Sprintf("KVCache.Append: K and V shape mismatch: %v vs %v", newK.shape, newV.shape))
	}
	if newK.Rank() < 3 {
		panic(fmt.Sprintf("KVCache.Append: K rank=%d, want >= 3 [batch, heads, seq, headDim]", newK.Rank()))
	}
	seqAxis := newK.Rank() - 2 // [batch, heads, seq, headDim] -> seq is at index 2
	headDim := newK.shape[newK.Rank()-1]
	otherShape := newK.shape[:seqAxis]
	newLen := newK.shape[seqAxis]

	cur := c.layers[layerIdx]
	if cur.K == nil {
		// First append: allocate the initial buffer.
		// We pre-allocate a small amount (1) and let Append
		// grow as needed. The caller is expected to call
		// Reset between sequences.
		combinedShape := append(append([]int{}, otherShape...), newLen, headDim)
		var newKT, newVT *Tensor
		if c.dtype == BFloat16 {
			newKT = ZerosBF16(combinedShape...)
			newVT = ZerosBF16(combinedShape...)
		} else {
			newKT = ZerosF32(combinedShape...)
			newVT = ZerosF32(combinedShape...)
		}
		// Copy newK/newV into the new buffers.
		copyInto(newKT, newK, 0, newLen)
		copyInto(newVT, newV, 0, newLen)
		c.layers[layerIdx] = layerKV{K: newKT, V: newVT, total: newLen}
		return
	}

	// Subsequent append: grow by newLen. The K/V are stored
	// row-major, so a simple slice growth is enough.
	cur.total += newLen
	grownShape := append(append([]int{}, otherShape...), cur.total, headDim)
	var newKT, newVT *Tensor
	if c.dtype == BFloat16 {
		newKT = ZerosBF16(grownShape...)
		newVT = ZerosBF16(grownShape...)
	} else {
		newKT = ZerosF32(grownShape...)
		newVT = ZerosF32(grownShape...)
	}
	// Copy the old K/V (rows 0..oldTotal) into the new buffers.
	copyInto(newKT, cur.K, 0, cur.total-newLen)
	copyInto(newVT, cur.V, 0, cur.total-newLen)
	// Copy the new K/V (rows oldTotal..cur.total) into the new buffers.
	copyInto(newKT, newK, cur.total-newLen, newLen)
	copyInto(newVT, newV, cur.total-newLen, newLen)
	// Free the old buffers.
	if c.dtype == BFloat16 {
		alloc.Free(cur.K.bf16)
		alloc.Free(cur.V.bf16)
	} else {
		alloc.Free(cur.K.f32)
		alloc.Free(cur.V.f32)
	}
	c.layers[layerIdx] = layerKV{K: newKT, V: newVT, total: cur.total}
}

// View returns the full K and V tensors for layerIdx. The
// returned tensors are borrowed (still owned by the cache);
// the caller must not free them. The seq dim is total.
func (c *KVCache) View(layerIdx int) (k, v *Tensor) {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		panic(fmt.Sprintf("KVCache.View: layerIdx=%d out of range", layerIdx))
	}
	cur := c.layers[layerIdx]
	if cur.K == nil {
		return nil, nil
	}
	return cur.K, cur.V
}

// TotalLen returns the current sequence length stored for
// layerIdx. 0 if no Append has happened for that layer yet.
func (c *KVCache) TotalLen(layerIdx int) int {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		return 0
	}
	return c.layers[layerIdx].total
}

// sameShape reports whether two shape slices are equal.
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// copyInto copies `src` into `dst`'s seq-dim starting at
// `startRow` for `seqLen` rows. Both tensors must have the
// same per-row element count and the same shape on every
// axis except possibly the seq axis.
func copyInto(dst, src *Tensor, startRow, seqLen int) {
	srcData, _ := src.ToF32()
	if dst.dtype == BFloat16 {
		// Cast to bf16 in place.
		seqAxis := src.Rank() - 2
		headDim := src.shape[src.Rank()-1]
		otherDims := src.shape[:seqAxis]
		// Walk rows.
		rowSize := headDim
		for i := 1; i < seqAxis; i++ {
			rowSize *= otherDims[i]
		}
		// The dst.bf16 is laid out as a single contiguous
		// slice. Walk it in rowSize chunks; convert each
		// chunk from float32 to bf16.
		for r := 0; r < seqLen; r++ {
			dstRow := (startRow + r) * rowSize
			srcRow := r * rowSize
			for j := 0; j < rowSize; j++ {
				dst.bf16[dstRow+j] = BF16FromF32(srcData[srcRow+j])
			}
		}
	} else {
		// Float32: direct copy. We assume the dst buffer
		// is large enough.
		seqAxis := src.Rank() - 2
		headDim := src.shape[src.Rank()-1]
		otherDims := src.shape[:seqAxis]
		rowSize := headDim
		for i := 1; i < seqAxis; i++ {
			rowSize *= otherDims[i]
		}
		for r := 0; r < seqLen; r++ {
			dstRow := (startRow + r) * rowSize
			srcRow := r * rowSize
			copy(dst.f32[dstRow:dstRow+rowSize], srcData[srcRow:srcRow+rowSize])
		}
	}
}
