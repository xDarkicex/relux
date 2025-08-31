package compute

import (
	"sync"
)

// TensorPool manages reusable float64 slices to reduce GC pressure
type TensorPool struct {
	pools   map[int]*sync.Pool // Pools by size
	mutex   sync.RWMutex
	maxSize int
}

var globalTensorPool = NewTensorPool(10000) // Max 10K element tensors

// NewTensorPool creates a new tensor memory pool
func NewTensorPool(maxTensorSize int) *TensorPool {
	return &TensorPool{
		pools:   make(map[int]*sync.Pool),
		maxSize: maxTensorSize,
	}
}

// Get retrieves a tensor of the specified size from the pool
func (tp *TensorPool) Get(size int) []float64 {
	if size <= 0 || size > tp.maxSize {
		// Too large for pooling, allocate directly
		return make([]float64, size)
	}

	// Round up to next power of 2 for better pooling efficiency
	poolSize := nextPowerOf2(size)

	tp.mutex.RLock()
	pool, exists := tp.pools[poolSize]
	tp.mutex.RUnlock()

	if !exists {
		tp.mutex.Lock()
		// Double-check pattern
		if pool, exists = tp.pools[poolSize]; !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return make([]float64, poolSize)
				},
			}
			tp.pools[poolSize] = pool
		}
		tp.mutex.Unlock()
	}

	slice := pool.Get().([]float64)
	return slice[:size] // Return slice with correct size
}

// Put returns a tensor to the pool for reuse
func (tp *TensorPool) Put(tensor []float64) {
	size := cap(tensor) // Use capacity, not length
	if size <= 0 || size > tp.maxSize {
		return // Don't pool oversized tensors
	}

	poolSize := nextPowerOf2(size)

	tp.mutex.RLock()
	pool, exists := tp.pools[poolSize]
	tp.mutex.RUnlock()

	if exists {
		// Clear the slice before returning to pool
		for i := range tensor[:cap(tensor)] {
			tensor[i] = 0
		}
		pool.Put(tensor[:cap(tensor)]) // Put full capacity slice
	}
}

// nextPowerOf2 returns the next power of 2 >= n
func nextPowerOf2(n int) int {
	if n <= 1 {
		return 1
	}

	// Find the highest bit set
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++

	return n
}

// PooledTensor wraps a pooled tensor with automatic cleanup
type PooledTensor struct {
	data []float64
	pool *TensorPool
}

// NewPooledTensor creates a tensor that will be automatically returned to pool
func NewPooledTensor(size int) *PooledTensor {
	return &PooledTensor{
		data: globalTensorPool.Get(size),
		pool: globalTensorPool,
	}
}

func (pt *PooledTensor) Data() []float64 {
	return pt.data
}

func (pt *PooledTensor) Release() {
	if pt.data != nil && pt.pool != nil {
		pt.pool.Put(pt.data)
		pt.data = nil
		pt.pool = nil
	}
}

// Auto-cleanup when GC'd (backup mechanism)
func (pt *PooledTensor) finalize() {
	pt.Release()
}
