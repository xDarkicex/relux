// Package alloc exposes typed off-heap allocation backed by the
// github.com/xDarkicex/memory package. The intent is to keep hot-path
// buffers (layer weights, gradient accumulators, intermediate activations)
// out of the Go GC's reach so that training and inference don't pay
// stop-the-world pauses proportional to model size.
//
// The backing allocator is a memory.ShardedFreeList. ShardedFreeList
// is the fastest of memory's per-object allocator types: lock-free
// Treiber stacks sharded by goroutine, with per-object Deallocate so
// callers may return slots to the freelist for reuse. Slices are
// bucketed by size (rounded up to the next power of two) so a small
// handful of freelists serves the whole range of allocation sizes the
// framework encounters.
//
// The slice header (data, len, cap) still lives on the Go heap, so
// the GC can trace the slice itself; only the data behind it is
// mmap'd. This means callers may freely pass these slices to cgo
// kernels (Metal, CUDA) without violating the cgo pointer rules.
//
// On any allocation failure (freelist exhausted, mmap denied) the
// helpers fall back to the standard Go heap. This is intentional:
// the off-heap path is an optimization, not a correctness requirement.
package alloc

import (
	"sync"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	// minSlotSize matches memory.ShardedFreeList's minimum (Hyaline
	// metadata: 44 bytes, padded to 48 for alignment).
	minSlotSize = 48

	// defaultPoolSize is the per-bucket memory budget. Lazily allocated
	// buckets only consume this much when they're actually used.
	defaultPoolSize = 16 * 1024 * 1024
)

var (
	freelists sync.Map // map[uint64]*memory.ShardedFreeList, keyed by slot size
	slotOwner sync.Map // map[uintptr]slotInfo, used by Free to look up the freelist
)

// slotInfo pairs a freelist with its slot size. ShardedFreeList does
// not expose SlotSize, so the alloc caller records the size at Allocate
// time for Free to use.
type slotInfo struct {
	fl       *memory.ShardedFreeList
	slotSize uint64
}

// bucketFor rounds size up to the next power of two, with a floor of
// minSlotSize. Power-of-two buckets mean N freelists cover the full
// range 48B..PoolSize with no size-class fragmentation.
func bucketFor(size uint64) uint64 {
	if size <= minSlotSize {
		return minSlotSize
	}
	p := uint64(minSlotSize)
	for p < size {
		p <<= 1
	}
	return p
}

func getFreelist(slotSize uint64) *memory.ShardedFreeList {
	if v, ok := freelists.Load(slotSize); ok {
		return v.(*memory.ShardedFreeList)
	}
	cfg := memory.DefaultFreeListConfig()
	cfg.SlotSize = slotSize
	cfg.PoolSize = defaultPoolSize
	fl, err := memory.NewShardedFreeList(cfg, 0) // 0 = default shard count
	if err != nil {
		return nil
	}
	actual, _ := freelists.LoadOrStore(slotSize, fl)
	return actual.(*memory.ShardedFreeList)
}

// Float64 returns a float64 slice of length n whose backing storage
// lives in an off-heap freelist. The returned slice is zero-initialized
// Float32 returns a float32 slice of length n, zero-initialized.
// The backing storage lives in the off-heap freelist; see
// [Float64] for semantics. Added for the transformer package,
// whose bfloat16-master / float32-active mixed precision needs
// a float32 pool.
func Float32(n int) []float32 {
	if n <= 0 {
		return nil
	}
	bytes := uint64(n) * 4
	slotSize := bucketFor(bytes)
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			s := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
			clear(s)
			return s
		}
	}
	out := make([]float32, n)
	clear(out)
	return out
}

// Float64 returns a float64 slice of length n whose backing storage
// lives in an off-heap freelist. The slice is zero-initialized
// (ShardedFreeList reuses slots, so leftover data is wiped on hand-out).
// Use [Free] to return the slice to the freelist for reuse.
func Float64(n int) []float64 {
	if n <= 0 {
		return nil
	}
	bytes := uint64(n) * 8
	slotSize := bucketFor(bytes)
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			s := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), n)
			clear(s) // zero-init the visible portion of the slot
			return s
		}
	}
	return make([]float64, n)
}

// ByteSlice returns a byte slice of length n, zero-initialized. See
// [Float64] for semantics.
func ByteSlice(n int) []byte {
	if n <= 0 {
		return nil
	}
	slotSize := bucketFor(uint64(n))
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			clear(buf)
			return buf
		}
	}
	return make([]byte, n)
}

// Uint16 returns a uint16 slice of length n, zero-initialized. The
// backing storage lives in the off-heap freelist; see [Float64] for
// semantics. Used by the transformer package for bfloat16 backings
// (each bfloat16 is one uint16 of mantissa-truncated float32 bits).
func Uint16(n int) []uint16 {
	if n <= 0 {
		return nil
	}
	slotSize := bucketFor(uint64(n) * 2)
	fl := getFreelist(slotSize)
	if fl != nil {
		if buf, err := fl.Allocate(); err == nil {
			addr := uintptr(unsafe.Pointer(&buf[0]))
			slotOwner.Store(addr, slotInfo{fl: fl, slotSize: slotSize})
			s := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[0])), n)
			clear(s)
			return s
		}
	}
	out := make([]uint16, n)
	clear(out)
	return out
}

// Free returns a slice previously obtained from [Float64] (or
// [ByteSlice]) to its freelist for reuse. Calling Free on a slice
// that was not produced by this package, or that has already been
// freed, is a no-op. After Free, the slice must not be used; the
// underlying memory may be handed out to a subsequent allocation.
func Free[T any](s []T) {
	if len(s) == 0 {
		return
	}
	addr := uintptr(unsafe.Pointer(&s[0]))
	v, ok := slotOwner.LoadAndDelete(addr)
	if !ok {
		return
	}
	info := v.(slotInfo)
	slot := unsafe.Slice((*byte)(unsafe.Pointer(addr)), info.slotSize)
	_ = info.fl.Deallocate(slot)
}

// Finalize releases every off-heap freelist this package has created.
// After Finalize, subsequent calls to Float64/ByteSlice re-initialize
// the freelists on first use. Call at process shutdown if you want
// deterministic memory release; otherwise the OS reclaims mmap'd
// pages on exit.
func Finalize() {
	freelists.Range(func(k, v any) bool {
		_ = v.(*memory.ShardedFreeList).Free()
		freelists.Delete(k)
		return true
	})
	slotOwner.Clear()
}
