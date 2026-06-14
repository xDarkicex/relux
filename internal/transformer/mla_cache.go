package transformer

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/xDarkicex/memory"
	"github.com/xDarkicex/relux/internal/alloc"
)

// MLACache holds the compressed KV state for Multi-head Latent Attention.
// Instead of storing full [numHeads, headDim] K/V per position, it stores
// the low-rank latent c^KV (dC dims) + the decoupled RoPE key k^R (dHR dims).
//
// Per-token cache: dC + dHR float32s (vs MHA's 2 × numHeads × headDim).
//
// Two allocation modes:
//   - Alloc-backed (maxLen>0): pre-allocates off-heap via alloc.Float32.
//   - File-backed: mmap'd file via memory.MmapFile.
type MLACache struct {
	layers   []mlaLayerKV
	maxLen   int
	dC       int
	dHR      int
	filePath string
	mmapData []byte
}

type mlaLayerKV struct {
	C_KV     *Tensor // [1, maxLen, dC] compressed KV latent
	K_R      *Tensor // [1, maxLen, dHR] RoPE key (post-rotation)
	total    int     // number of positions stored
	capacity int     // size of pre-allocated buffer
}

// NewMLACache constructs a pre-allocated MLACache backed by alloc.Float32.
func NewMLACache(numLayers int, maxLen int, dC int, dHR int) *MLACache {
	if numLayers <= 0 {
		panic(fmt.Sprintf("NewMLACache: numLayers=%d, must be > 0", numLayers))
	}
	if maxLen <= 0 {
		panic(fmt.Sprintf("NewMLACache: maxLen=%d, must be > 0", maxLen))
	}
	layers := make([]mlaLayerKV, numLayers)
	for i := range layers {
		layers[i] = mlaLayerKV{
			C_KV:     ZerosF32(1, maxLen, dC),
			K_R:      ZerosF32(1, maxLen, dHR),
			capacity: maxLen,
		}
	}
	return &MLACache{
		layers: layers,
		maxLen: maxLen,
		dC:     dC,
		dHR:    dHR,
	}
}

// NewMLACacheFile constructs a pre-allocated MLACache backed by a file-mmap'd
// region. The file is created (or truncated) at filePath with the required size.
func NewMLACacheFile(numLayers int, maxLen int, dC int, dHR int, filePath string) (*MLACache, error) {
	if numLayers <= 0 || maxLen <= 0 {
		return nil, fmt.Errorf("NewMLACacheFile: numLayers=%d maxLen=%d, both must be > 0", numLayers, maxLen)
	}
	// Per layer: C_KV (maxLen*dC) + K_R (maxLen*dHR), both float32 (4 bytes each).
	perLayer := uint64(maxLen) * uint64(dC+dHR) * 4
	totalBytes := uint64(numLayers) * perLayer

	f, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("NewMLACacheFile: open: %w", err)
	}
	defer f.Close()

	if err := f.Truncate(int64(totalBytes)); err != nil {
		return nil, fmt.Errorf("NewMLACacheFile: truncate: %w", err)
	}

	data, err := memory.MmapFile(int(f.Fd()), 0, int(totalBytes), true)
	if err != nil {
		return nil, fmt.Errorf("NewMLACacheFile: mmap: %w", err)
	}

	layers := make([]mlaLayerKV, numLayers)
	basePtr := unsafe.Pointer(&data[0])

	for i := range layers {
		// C_KV: [1, maxLen, dC]
		cSize := maxLen * dC
		cSlice := unsafe.Slice((*float32)(basePtr), cSize)
		basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(cSize*4))
		cTensor := &Tensor{shape: []int{1, maxLen, dC}, dtype: Float32, f32: cSlice}

		// K_R: [1, maxLen, dHR]
		krSize := maxLen * dHR
		krSlice := unsafe.Slice((*float32)(basePtr), krSize)
		basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(krSize*4))
		krTensor := &Tensor{shape: []int{1, maxLen, dHR}, dtype: Float32, f32: krSlice}

		layers[i] = mlaLayerKV{C_KV: cTensor, K_R: krTensor, capacity: maxLen}
	}

	return &MLACache{
		layers:   layers,
		maxLen:   maxLen,
		dC:       dC,
		dHR:      dHR,
		filePath: filePath,
		mmapData: data,
	}, nil
}

// Reset clears all stored positions. For pre-allocated caches only the cursor
// is reset (buffers remain allocated). For file-backed caches the mmap is
// unmapped and the layers are re-created from the file on next use.
func (c *MLACache) Reset() {
	for i := range c.layers {
		if c.layers[i].capacity > 0 {
			c.layers[i].total = 0
			c.zeroLayer(i)
		}
	}
	if c.mmapData != nil {
		memory.Munmap(c.mmapData)
		c.mmapData = nil
		for i := range c.layers {
			c.layers[i].C_KV = nil
			c.layers[i].K_R = nil
		}
	}
}

// zeroLayer zero-initializes layer i's buffers.
func (c *MLACache) zeroLayer(i int) {
	cur := c.layers[i]
	if cur.C_KV != nil {
		for j := range cur.C_KV.f32 {
			cur.C_KV.f32[j] = 0
		}
	}
	if cur.K_R != nil {
		for j := range cur.K_R.f32 {
			cur.K_R.f32[j] = 0
		}
	}
}

// Append copies newC and newKR into the cache at the current cursor position.
// newC is [1, newLen, dC], newKR is [1, newLen, dHR].
func (c *MLACache) Append(layerIdx int, newC, newKR *Tensor) {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		panic(fmt.Sprintf("MLACache.Append: layerIdx=%d out of range [0, %d)", layerIdx, len(c.layers)))
	}
	cur := c.layers[layerIdx]

	if cur.C_KV == nil && c.mmapData == nil && c.filePath != "" {
		c.reinitFromFile()
		cur = c.layers[layerIdx]
	}

	newLen := newC.shape[newC.Rank()-2]
	if cur.total+newLen > cur.capacity {
		panic(fmt.Sprintf("MLACache.Append: layer %d overflow: total=%d + newLen=%d > capacity=%d",
			layerIdx, cur.total, newLen, cur.capacity))
	}

	newCData, _ := newC.ToF32()
	newKRData, _ := newKR.ToF32()
	copy(cur.C_KV.f32[cur.total*c.dC:], newCData)
	copy(cur.K_R.f32[cur.total*c.dHR:], newKRData)
	if newC.dtype != Float32 {
		alloc.Free(newCData)
	}
	if newKR.dtype != Float32 {
		alloc.Free(newKRData)
	}

	cur.total += newLen
	c.layers[layerIdx] = cur
}

// reinitFromFile re-mmaps the file-backed buffers after Reset.
func (c *MLACache) reinitFromFile() {
	if c.filePath == "" || c.mmapData != nil {
		return
	}
	perLayer := uint64(c.maxLen) * uint64(c.dC+c.dHR) * 4
	totalBytes := int(uint64(len(c.layers)) * perLayer)

	f, err := os.Open(c.filePath)
	if err != nil {
		panic(fmt.Sprintf("MLACache.reinitFromFile: open %s: %v", c.filePath, err))
	}
	defer f.Close()

	data, err := memory.MmapFile(int(f.Fd()), 0, totalBytes, true)
	if err != nil {
		panic(fmt.Sprintf("MLACache.reinitFromFile: mmap: %v", err))
	}
	c.mmapData = data

	basePtr := unsafe.Pointer(&data[0])
	for i := range c.layers {
		cSize := c.maxLen * c.dC
		cSlice := unsafe.Slice((*float32)(basePtr), cSize)
		basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(cSize*4))
		cTensor := &Tensor{shape: []int{1, c.maxLen, c.dC}, dtype: Float32, f32: cSlice}

		krSize := c.maxLen * c.dHR
		krSlice := unsafe.Slice((*float32)(basePtr), krSize)
		basePtr = unsafe.Pointer(uintptr(basePtr) + uintptr(krSize*4))
		krTensor := &Tensor{shape: []int{1, c.maxLen, c.dHR}, dtype: Float32, f32: krSlice}

		c.layers[i] = mlaLayerKV{C_KV: cTensor, K_R: krTensor, capacity: c.maxLen}
	}
}

// View returns sub-views of the cached C_KV and K_R for layerIdx,
// shaped [1, total, dC] and [1, total, dHR] respectively.
func (c *MLACache) View(layerIdx int) (cKV, kR *Tensor) {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		panic(fmt.Sprintf("MLACache.View: layerIdx=%d out of range", layerIdx))
	}
	cur := c.layers[layerIdx]
	if cur.C_KV == nil {
		return nil, nil
	}
	viewC := &Tensor{shape: []int{1, cur.total, c.dC}, dtype: Float32, f32: cur.C_KV.f32}
	viewKR := &Tensor{shape: []int{1, cur.total, c.dHR}, dtype: Float32, f32: cur.K_R.f32}
	return viewC, viewKR
}

// TotalLen returns the current sequence length stored for layerIdx.
func (c *MLACache) TotalLen(layerIdx int) int {
	if layerIdx < 0 || layerIdx >= len(c.layers) {
		return 0
	}
	return c.layers[layerIdx].total
}
