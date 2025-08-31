package compute

// ComputeBackend provides hardware-accelerated tensor operations
// Framework-agnostic interface that works with any backend
type ComputeBackend interface {
	// Core matrix operations (80% of ML compute time)
	MatMul(A, B [][]float64) ([][]float64, error)

	// Vector operations for bias addition and element-wise ops
	VectorAdd(A, B []float64) ([]float64, error)
	VectorSub(A, B []float64) ([]float64, error)
	VectorMul(A, B []float64) ([]float64, error) // Element-wise

	// Activation functions (critical for training performance)
	ActivationFunc(name string, x []float64) ([]float64, error)

	// Backend information
	Name() string
	Available() bool
	DeviceInfo() string
	Close() error
}

// BackendType represents different compute backend types
type BackendType int

const (
	BackendAuto   BackendType = iota // Auto-detect best backend
	BackendNative                    // Pure Go implementation
	BackendRnxa                      // rnxa hardware acceleration
)
