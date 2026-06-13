package compute

// ComputeBackend provides hardware-accelerated tensor operations.
//
// Float64 methods (MatMul, VectorAdd, etc.) are the legacy API for
// the pre-bf16 code path. They remain for backward compat with the
// existing MLP framework's introspection and benchmarking paths,
// but they are no longer on the hot compute path.
//
// Float32 methods are the active compute path for the bf16 refactor.
// Weights are passed as float32 (already widened from bf16 by the
// caller), activations and results are float32.
type ComputeBackend interface {
	// Legacy float64 API — kept for compat / benchmarks.
	MatMul(A, B [][]float64) ([][]float64, error)
	VectorAdd(A, B []float64) ([]float64, error)
	VectorSub(A, B []float64) ([]float64, error)
	VectorMul(A, B []float64) ([]float64, error)
	ActivationFunc(name string, x []float64) ([]float64, error)
	BatchMatMul(matrices [][][]float64) ([][][]float64, error)
	ForwardBatch(inputs [][]float64, weights [][]float64, biases []float64, activation string) ([][]float64, error)

	// Active compute path — float32, for bf16 mixed precision.
	//
	// MatMulFloat32 computes C = A @ B where A is [M, K] and B
	// is [K, N], both row-major float32. C is [M, N] row-major
	// float32. The caller is responsible for widening bf16
	// weights to float32 before calling this method.
	MatMulFloat32(A, B []float32, M, K, N int) ([]float32, error)

	// Performance configuration access.
	GetPerformanceConfig() *PerformanceConfig
	ShouldUseGPUForMatMul(M, N, K int) bool
	ShouldUseGPUForActivation(size int) bool

	// Backend information.
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
