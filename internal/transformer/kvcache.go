package transformer

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/xDarkicex/memory"
	"github.com/xDarkicex/relux/internal/alloc"
)

// KVCache holds the per-layer K and V state for autoregressive
// generation. During decode the new K/V (for the just-generated
// token) is appended to the cache; the next attention call
// reads the full sequence from K[0:totalLen] and V[0:totalLen].
//
// Storage is bfloat16 (the active dtype) by default; the
// float32 path is used for training and roundtrip tests.
//
// Two allocation modes:
//
//   - Legacy (maxLen=0): grows on every Append. NewKVCache.
//   - Sized (maxLen>0): pre-allocates the full buffer at
//     construction. Append is cursor-based (O(1) per step).
//     NewKVCacheSized (alloc-backed) or NewKVCacheFile
//     (mmap/file-backed).
//
// For a 128k-token / 32-layer / 8-KV-head / 128-headDim / bf16:
//
//	2 × 32 × 128k × 8 × 128 × 2 = 16 GB total
//
// When file-backed the OS pages cold slots to SSD and keeps
// hot slots in RAM transparently.
type KVCache struct {
	layers     []layerKV
	dtype      DType
	maxLen     int    // 0 = grow-as-needed; >0 = pre-allocated
	numKVHeads int    // cached for View reshaping (sized path)
	headDim    int    // cached for View reshaping (sized path)
	filePath   string // non-empty = mmap-backed
	mmapData   []byte // raw mmap'd backing (set for file-backed path)
}

type layerKV struct {
	K        *Tensor
	V        *Tensor
	total    int // number of positions stored (seq dim)
	capacity int // size of pre-allocated buffer (0 = legacy grow path)
}

// NewKVCache constructs a legacy KVCache (grows on every Append).
// For pre-allocation use NewKVCacheSized or NewKVCacheFile.
func NewKVCache(numLayers int, dtype DType) *KVCache {
	if numLayers <= 0 {
		panic(fmt.Sprintf("NewKVCache: numLayers=%d, must be > 0", numLayers))
	}
	return &KVCache{
		layers: make([]layerKV, numLayers),
		dtype:  dtype,
	}
}

// NewKVCacheSized constructs a pre-allocated KVCache. K/V buffers
// of shape [1, numKVHeads, maxLen, headDim] are allocated upfront
// (off-heap via the alloc package). Append is cursor-based.
func NewKVCacheSized(numLayers int, maxLen int, numKVHeads int, headDim int, dtype DType) *KVCache {
	if numLayers <= 0 {
		panic(fmt.Sprintf("NewKVCacheSized: numLayers=%d, must be > 0", numLayers))
	}
	if maxLen <= 0 {
		panic(fmt.Sprintf("NewKVCacheSized: maxLen=%d, must be > 0", maxLen))
	}
	shape := []int{1, numKVHeads, maxLen, headDim}
	layers := make([]layerKV, numLayers)
	for i := range layers {
		var kt, vt *Tensor
		if dtype == BFloat16 {
			kt = ZerosBF16(shape...)
			vt = ZerosBF16(shape...)
		} else {
			kt = ZerosF32(shape...)
			vt = ZerosF32(shape...)
		}
		layers[i] = layerKV{K: kt, V: vt, capacity: maxLen}
	}
	return &KVCache{
		layers:     layers,
		dtype:      dtype,
		maxLen:     maxLen,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}
}

// NewKVCacheFile constructs a pre-allocated KVCache backed by a
// file-mmap'd region. The file is created (or truncated) at
// filePath with the required size. The backing is mapped with
// MAP_SHARED so writes are durable (useful for crash recovery
// if the caller checkpoints the file path).
//
// The caller owns the file lifecycle: the file is NOT deleted
// on Reset or on GC. Call os.Remove(filePath) manually when
// done.
func NewKVCacheFile(numLayers int, maxLen int, numKVHeads int, headDim int, dtype DType, filePath string) (*KVCache, error) {
	if numLayers <= 0 || maxLen <= 0 {
		return nil, fmt.Errorf("NewKVCacheFile: numLayers=%d maxLen=%d, both must be > 0", numLayers, maxLen)
	}
	elemBytes := 4
	if dtype == BFloat16 {
		elemBytes = 2
	}
	// 2 (K+V) × numLayers × batch(1) × numKVHeads × maxLen × headDim × elemBytes
	perLayer := uint64(1) * uint64(numKVHeads) * uint64(maxLen) * uint64(headDim) * uint64(elemBytes)
	totalBytes := uint64(numLayers) * 2 * perLayer

	f, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("NewKVCacheFile: open: %w", err)
	}
	defer f.Close()

	if err := f.Truncate(int64(totalBytes)); err != nil {
		return nil, fmt.Errorf("NewKVCacheFile: truncate: %w", err)
	}

	data, err := memory.MmapFile(int(f.Fd()), 0, int(totalBytes), true)
	if err != nil {
		return nil, fmt.Errorf("NewKVCacheFile: mmap: %w", err)
	}

	shape := []int{1, numKVHeads, maxLen, headDim}
	size := maxLen * numKVHeads * headDim // element count
	layers := make([]layerKV, numLayers)
	basePtr := unsafe.Pointer(&data[0])

	for i := range layers {
		// Cast mmap byte region to typed slice.
		var kt, vt *Tensor
		if dtype == BFloat16 {
			kSlice := unsafe.Slice((*uint16)(basePtr), size)
			basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(size*2))
			vSlice := unsafe.Slice((*uint16)(basePtr), size)
			basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(size*2))
			kt = &Tensor{shape: append([]int{}, shape...), dtype: dtype, bf16: kSlice}
			vt = &Tensor{shape: append([]int{}, shape...), dtype: dtype, bf16: vSlice}
		} else {
			kSlice := unsafe.Slice((*float32)(basePtr), size)
			basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(size*4))
			vSlice := unsafe.Slice((*float32)(basePtr), size)
			basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(size*4))
			kt = &Tensor{shape: append([]int{}, shape...), dtype: dtype, f32: kSlice}
			vt = &Tensor{shape: append([]int{}, shape...), dtype: dtype, f32: vSlice}
		}
		layers[i] = layerKV{K: kt, V: vt, capacity: maxLen}
	}

	return &KVCache{
		layers:     layers,
		dtype:      dtype,
		maxLen:     maxLen,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		filePath:   filePath,
		mmapData:   data,
	}, nil
}

// DType returns the active dtype of the cache.
func (c *KVCache) DType() DType { return c.dtype }

// Reset clears all stored K/V. For the legacy path the buffers
// are freed and set to nil. For the pre-allocated path only the
// cursor is reset (the buffers remain allocated).
func (c *KVCache) Reset() {
	for i := range c.layers {
		cur := c.layers[i]
		if cur.capacity > 0 {
			c.layers[i].total = 0
			c.zeroLayer(i, c.maxLen)
		} else {
			if cur.K != nil {
				alloc.Free(cur.K.f32)
				alloc.Free(cur.K.f64)
				alloc.Free(cur.K.bf16)
			}
			if cur.V != nil {
				alloc.Free(cur.V.f32)
				alloc.Free(cur.V.f64)
				alloc.Free(cur.V.bf16)
			}
			c.layers[i] = layerKV{}
		}
	}
	if c.mmapData != nil {
		memory.Munmap(c.mmapData)
		c.mmapData = nil
		for i := range c.layers {
			c.layers[i].K = nil
			c.layers[i].V = nil
		}
	}
}

// zeroLayer zero-initializes the first `total` tokens' worth of layer i.
func (c *KVCache) zeroLayer(i int, total int) {
	cur := c.layers[i]
	if cur.K == nil || total == 0 {
		return
	}
	totalElems := total * c.numKVHeads * c.headDim
	if c.dtype == BFloat16 {
		z := cur.K.bf16[:totalElems]
		for j := range z {
			z[j] = 0
		}
		z = cur.V.bf16[:totalElems]
		for j := range z {
			z[j] = 0
		}
	} else {
		z := cur.K.f32[:totalElems]
		for j := range z {
			z[j] = 0
		}
		z = cur.V.f32[:totalElems]
		for j := range z {
			z[j] = 0
		}
	}
}

// Append concatenates newK and newV onto layerIdx's K and V.
//
// In legacy mode (maxLen=0) the buffer grows on every call.
// In pre-allocated mode (maxLen>0) the new K/V is copied into
// the pre-allocated buffer at the cursor position — no
// re-allocation.
//
// newK and newV are NOT consumed; the caller still owns them.
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
	seqAxis := newK.Rank() - 2
	headDim := newK.shape[newK.Rank()-1]
	otherShape := newK.shape[:seqAxis]
	newLen := newK.shape[seqAxis]

	cur := c.layers[layerIdx]

	if cur.K == nil {
		if c.maxLen > 0 {
			c.reinitFromFile()
			cur = c.layers[layerIdx]
		}
		if cur.K == nil {
			// Legacy path: allocate with exactly newLen.
			combinedShape := append(append([]int{}, otherShape...), newLen, headDim)
			var newKT, newVT *Tensor
			if c.dtype == BFloat16 {
				newKT = ZerosBF16(combinedShape...)
				newVT = ZerosBF16(combinedShape...)
			} else {
				newKT = ZerosF32(combinedShape...)
				newVT = ZerosF32(combinedShape...)
			}
			copyInto(newKT, newK, 0, newLen)
			copyInto(newVT, newV, 0, newLen)
			c.layers[layerIdx] = layerKV{K: newKT, V: newVT, total: newLen}
			return
		}
	}

	if cur.capacity > 0 {
		// Pre-allocated: cursor copy.
		if cur.total+newLen > cur.capacity {
			panic(fmt.Sprintf("KVCache.Append: layer %d overflow: total=%d + newLen=%d > capacity=%d",
				layerIdx, cur.total, newLen, cur.capacity))
		}
		copyInto(cur.K, newK, cur.total, newLen)
		copyInto(cur.V, newV, cur.total, newLen)
		cur.total += newLen
		c.layers[layerIdx] = cur
		return
	}

	// Legacy grow path: allocate new, copy old + new, free old.
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
	copyInto(newKT, cur.K, 0, cur.total-newLen)
	copyInto(newVT, cur.V, 0, cur.total-newLen)
	copyInto(newKT, newK, cur.total-newLen, newLen)
	copyInto(newVT, newV, cur.total-newLen, newLen)
	if c.dtype == BFloat16 {
		alloc.Free(cur.K.bf16)
		alloc.Free(cur.V.bf16)
	} else {
		alloc.Free(cur.K.f32)
		alloc.Free(cur.V.f32)
	}
	c.layers[layerIdx] = layerKV{K: newKT, V: newVT, total: cur.total}
}

// reinitFromFile re-mmaps the file-backed K/V buffers after Reset.
func (c *KVCache) reinitFromFile() {
	if c.filePath == "" {
		return
	}
	elemBytes := 4
	if c.dtype == BFloat16 {
		elemBytes = 2
	}
	size := c.maxLen * c.numKVHeads * c.headDim // element count
	if c.mmapData == nil {
		totalBytes := len(c.layers) * 2 * size * elemBytes
		f, err := os.Open(c.filePath)
		if err != nil {
			panic(fmt.Sprintf("KVCache.reinitFromFile: open %s: %v", c.filePath, err))
		}
		defer f.Close()
		data, err := memory.MmapFile(int(f.Fd()), 0, totalBytes, true)
		if err != nil {
			panic(fmt.Sprintf("KVCache.reinitFromFile: mmap: %v", err))
		}
		c.mmapData = data
	}
	shape := []int{1, c.numKVHeads, c.maxLen, c.headDim}
	basePtr := unsafe.Pointer(&c.mmapData[0])
	for i := range c.layers {
		kStart := uintptr(i * 2 * size * elemBytes)
		vStart := kStart + uintptr(size*elemBytes)
		if c.dtype == BFloat16 {
			kSlice := unsafe.Slice((*uint16)(unsafe.Pointer(uintptr(basePtr)+kStart)), size)
			vSlice := unsafe.Slice((*uint16)(unsafe.Pointer(uintptr(basePtr)+vStart)), size)
			c.layers[i] = layerKV{K: &Tensor{shape: append([]int{}, shape...), dtype: c.dtype, bf16: kSlice}, V: &Tensor{shape: append([]int{}, shape...), dtype: c.dtype, bf16: vSlice}, capacity: c.maxLen}
		} else {
			kSlice := unsafe.Slice((*float32)(unsafe.Pointer(uintptr(basePtr)+kStart)), size)
			vSlice := unsafe.Slice((*float32)(unsafe.Pointer(uintptr(basePtr)+vStart)), size)
			c.layers[i] = layerKV{K: &Tensor{shape: append([]int{}, shape...), dtype: c.dtype, f32: kSlice}, V: &Tensor{shape: append([]int{}, shape...), dtype: c.dtype, f32: vSlice}, capacity: c.maxLen}
		}
	}
}

// View returns the K and V tensors for layerIdx.
// For pre-allocated caches the returned tensors are sub-views
// shaped [1, numKVHeads, total, headDim] (not the full capacity).
// The tensors are borrowed (still owned by the cache); the caller
// must not free them.
func (c *KVCache) View(layerIdx int) (k, v *Tensor) {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		panic(fmt.Sprintf("KVCache.View: layerIdx=%d out of range", layerIdx))
	}
	cur := c.layers[layerIdx]
	if cur.K == nil {
		return nil, nil
	}
	if cur.capacity > 0 {
		// Return a sub-view shaped [1, numKVHeads, cur.total, headDim].
		viewShape := append([]int{}, cur.K.shape...)
		viewShape[len(viewShape)-2] = cur.total
		viewK := &Tensor{shape: viewShape, dtype: cur.K.dtype, f32: cur.K.f32, bf16: cur.K.bf16}
		viewV := &Tensor{shape: viewShape, dtype: cur.V.dtype, f32: cur.V.f32, bf16: cur.V.bf16}
		return viewK, viewV
	}
	return cur.K, cur.V
}

// TotalLen returns the current sequence length stored for layerIdx.
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
			for j := 0; j < rowSize; j++ {
				dst.bf16[dstRow+j] = BF16FromF32(srcData[srcRow+j])
			}
		}
	} else {
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
