package compute

// ComputeBackend provides hardware-accelerated tensor operations
type ComputeBackend interface {
	// Core matrix operations
	MatMul(A, B [][]float64) ([][]float64, error)

	// Vector operations
	VectorAdd(A, B []float64) ([]float64, error)
	VectorSub(A, B []float64) ([]float64, error)
	VectorMul(A, B []float64) ([]float64, error)

	// Activation functions
	ActivationFunc(name string, x []float64) ([]float64, error)

	// Enhanced batch operations
	BatchMatMul(matrices [][][]float64) ([][][]float64, error)
	ForwardBatch(inputs [][]float64, weights [][]float64, biases []float64, activation string) ([][]float64, error)

	// Performance configuration access
	GetPerformanceConfig() *PerformanceConfig
	ShouldUseGPUForMatMul(M, N, K int) bool
	ShouldUseGPUForActivation(size int) bool

	// Backend information
	Name() string
	Available() bool
	DeviceInfo() string
	Close() error
}

// BackendType represents different compute backend types
type BackendType int

const (
	BackendAuto BackendType = iota
	BackendNative
	BackendRnxa
)
