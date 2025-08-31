package compute

import (
	"runtime"
	"sync"
)

// PerformanceConfig contains tunable parameters for optimal performance
type PerformanceConfig struct {
	// Matrix multiplication thresholds
	MatMulGPUThreshold int // Minimum matrix size for GPU acceleration
	MatMulCPUParallel  int // Minimum size for CPU parallelization

	// Vector operation thresholds
	VectorGPUThreshold     int // Minimum vector size for GPU
	ActivationGPUThreshold int // Minimum activation vector size for GPU

	// Memory management
	EnableMemoryPooling bool // Enable tensor memory pooling
	PoolSize            int  // Maximum pooled tensors

	// CPU optimization
	MaxCPUThreads int // Maximum CPU threads for parallel ops

	// Batch processing
	OptimalBatchSize     int // Sweet spot for batch operations
	MaxConcurrentBatches int // Maximum concurrent batch processing
}

var (
	defaultConfig = &PerformanceConfig{
		// Conservative defaults - will be tuned based on device
		MatMulGPUThreshold:     128,  // 128x128 matrices and larger
		MatMulCPUParallel:      64,   // Parallelize CPU for 64x64+
		VectorGPUThreshold:     1000, // 1000+ element vectors
		ActivationGPUThreshold: 500,  // 500+ activations

		EnableMemoryPooling: true,
		PoolSize:            100, // Pool up to 100 tensors

		MaxCPUThreads:        runtime.NumCPU(),
		OptimalBatchSize:     32, // Good balance for most networks
		MaxConcurrentBatches: runtime.NumCPU(),
	}

	configMutex  sync.RWMutex
	globalConfig *PerformanceConfig
)

func init() {
	globalConfig = &PerformanceConfig{}
	*globalConfig = *defaultConfig // Copy defaults

	// Auto-tune based on hardware
	autoTuneConfig()
}

// GetPerformanceConfig returns current configuration (thread-safe)
func GetPerformanceConfig() *PerformanceConfig {
	configMutex.RLock()
	defer configMutex.RUnlock()

	config := &PerformanceConfig{}
	*config = *globalConfig // Return copy
	return config
}

// UpdatePerformanceConfig allows runtime tuning
func UpdatePerformanceConfig(updater func(*PerformanceConfig)) {
	configMutex.Lock()
	defer configMutex.Unlock()

	updater(globalConfig)
}

// autoTuneConfig adjusts defaults based on available hardware
func autoTuneConfig() {
	// Detect available backends and adjust accordingly
	backends := GetAvailableBackends()

	// If rnxa (GPU) is available, optimize for it
	for _, backend := range backends {
		if backend != "native (Pure Go)" {
			// GPU available - lower thresholds for better utilization
			globalConfig.MatMulGPUThreshold = 64  // Smaller threshold
			globalConfig.VectorGPUThreshold = 500 // More aggressive
			globalConfig.ActivationGPUThreshold = 250
			globalConfig.OptimalBatchSize = 64 // Larger batches for GPU
			break
		}
	}

	// Adjust for available CPU cores
	cores := runtime.NumCPU()
	if cores >= 8 {
		globalConfig.MatMulCPUParallel = 32 // Lower threshold for high-core systems
		globalConfig.MaxConcurrentBatches = cores
	}
}

// ShouldUseGPU returns whether GPU should be used for given operation size
func (c *PerformanceConfig) ShouldUseGPUForMatMul(M, N, K int) bool {
	// Use product of dimensions to determine complexity
	complexity := M * N * K
	threshold := c.MatMulGPUThreshold * c.MatMulGPUThreshold * c.MatMulGPUThreshold
	return complexity >= threshold
}

func (c *PerformanceConfig) ShouldUseGPUForVector(size int) bool {
	return size >= c.VectorGPUThreshold
}

func (c *PerformanceConfig) ShouldUseGPUForActivation(size int) bool {
	return size >= c.ActivationGPUThreshold
}
