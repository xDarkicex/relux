//go:build rnxa
// +build rnxa

package compute

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/rnxa"
)

// EnhancedRnxaBackend provides optimized batch operations and memory pooling
type enhancedRnxaBackend struct {
	*rnxaBackend // Embed base implementation

	config     *PerformanceConfig
	tensorPool *TensorPool

	// Batch processing
	batchMutex sync.Mutex
	batchJobs  chan batchJob
	batchPool  sync.Pool
}

type batchJob struct {
	operation string
	matrices  [][][]float64 // Multiple matrix pairs
	vectors   [][]float64   // Multiple vectors
	result    chan batchResult
}

type batchResult struct {
	matrices [][][]float64
	vectors  [][]float64
	err      error
}

func newEnhancedRnxaBackend() (ComputeBackend, error) {
	base, err := newRnxaBackend()
	if err != nil {
		return nil, err
	}

	baseRnxa := base.(*rnxaBackend)

	enhanced := &enhancedRnxaBackend{
		rnxaBackend: baseRnxa,
		config:      GetPerformanceConfig(),
		tensorPool:  NewTensorPool(50000), // Larger pool for GPU operations
	}

	// Initialize batch processing workers
	enhanced.initBatchWorkers()

	return enhanced, nil
}

func (e *enhancedRnxaBackend) initBatchWorkers() {
	workerCount := e.config.MaxConcurrentBatches
	e.batchJobs = make(chan batchJob, workerCount*2)

	// Start batch processing workers
	for i := 0; i < workerCount; i++ {
		go e.batchWorker()
	}
}

func (e *enhancedRnxaBackend) batchWorker() {
	for job := range e.batchJobs {
		result := batchResult{}

		switch job.operation {
		case "matmul":
			result.matrices, result.err = e.processBatchMatMul(job.matrices)
		case "vector_add":
			result.vectors, result.err = e.processBatchVectorOp(job.vectors, "add")
		default:
			result.err = fmt.Errorf("unsupported batch operation: %s", job.operation)
		}

		job.result <- result
		close(job.result)
	}
}

// Enhanced MatMul with intelligent switching and memory pooling
func (e *enhancedRnxaBackend) MatMul(A, B [][]float64) ([][]float64, error) {
	if len(A) == 0 || len(B) == 0 {
		return nil, fmt.Errorf("empty matrices")
	}
	if len(A[0]) != len(B) {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}

	M, K, N := len(A), len(A[0]), len(B[0])

	// Intelligent switching based on complexity and configuration
	if e.config.ShouldUseGPUForMatMul(M, N, K) {
		// Use GPU acceleration for large matrices
		return e.gpuMatMul(A, B)
	} else if M*N*K >= e.config.MatMulCPUParallel*e.config.MatMulCPUParallel*e.config.MatMulCPUParallel {
		// Use parallel CPU for medium matrices
		return e.parallelCPUMatMul(A, B)
	} else {
		// Use simple CPU for small matrices
		return e.rnxaBackend.MatMul(A, B)
	}
}

func (e *enhancedRnxaBackend) gpuMatMul(A, B [][]float64) ([][]float64, error) {
	// Convert with memory pooling
	tensorA := e.convertMatrixToTensorPooled(A)
	defer e.releaseTensor(tensorA)

	tensorB := e.convertMatrixToTensorPooled(B)
	defer e.releaseTensor(tensorB)

	// Hardware-accelerated computation
	result, err := e.engine.MatMul(e.ctx, tensorA, tensorB)
	if err != nil {
		return nil, fmt.Errorf("GPU MatMul failed: %w", err)
	}

	return convertTensorToMatrix(result), nil
}

func (e *enhancedRnxaBackend) parallelCPUMatMul(A, B [][]float64) ([][]float64, error) {
	M, K, N := len(A), len(A[0]), len(B[0])

	// Use pooled memory for result
	resultData := e.tensorPool.Get(M * N)
	defer e.tensorPool.Put(resultData)

	// Create result matrix view
	C := make([][]float64, M)
	for i := range C {
		C[i] = resultData[i*N : (i+1)*N]
	}

	// Parallel computation using goroutines
	numWorkers := runtime.NumCPU()
	if numWorkers > M {
		numWorkers = M
	}

	var wg sync.WaitGroup
	rowsPerWorker := M / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		wg.Add(1)
		go func(startRow, endRow int) {
			defer wg.Done()

			for i := startRow; i < endRow; i++ {
				for j := 0; j < N; j++ {
					for k := 0; k < K; k++ {
						C[i][j] += A[i][k] * B[k][j]
					}
				}
			}
		}(worker*rowsPerWorker, (worker+1)*rowsPerWorker)
	}

	// Handle remaining rows
	if remainder := M % numWorkers; remainder > 0 {
		startRow := numWorkers * rowsPerWorker
		for i := startRow; i < M; i++ {
			for j := 0; j < N; j++ {
				for k := 0; k < K; k++ {
					C[i][j] += A[i][k] * B[k][j]
				}
			}
		}
	}

	wg.Wait()

	// Copy result to properly sized matrix (since we used pooled memory)
	result := make([][]float64, M)
	for i := range result {
		result[i] = make([]float64, N)
		copy(result[i], C[i])
	}

	return result, nil
}

// BatchMatMul processes multiple matrix multiplications in parallel
func (e *enhancedRnxaBackend) BatchMatMul(matrices [][][]float64) ([][][]float64, error) {
	if len(matrices)%2 != 0 {
		return nil, fmt.Errorf("batch matmul requires even number of matrices (pairs)")
	}

	batchSize := e.config.OptimalBatchSize
	results := make([][][]float64, len(matrices)/2)

	// Process in batches for optimal GPU utilization
	for i := 0; i < len(matrices); i += batchSize * 2 {
		end := i + batchSize*2
		if end > len(matrices) {
			end = len(matrices)
		}

		batchMatrices := matrices[i:end]

		// Submit batch job
		job := batchJob{
			operation: "matmul",
			matrices:  batchMatrices,
			result:    make(chan batchResult, 1),
		}

		e.batchJobs <- job
		result := <-job.result

		if result.err != nil {
			return nil, result.err
		}

		// Copy results
		copy(results[i/2:], result.matrices)
	}

	return results, nil
}

func (e *enhancedRnxaBackend) processBatchMatMul(matrices [][][]float64) ([][][]float64, error) {
	results := make([][][]float64, len(matrices)/2)

	for i := 0; i < len(matrices); i += 2 {
		A, B := matrices[i], matrices[i+1]
		result, err := e.MatMul(A, B)
		if err != nil {
			return nil, err
		}
		results[i/2] = result
	}

	return results, nil
}

func (e *enhancedRnxaBackend) convertMatrixToTensorPooled(matrix [][]float64) *rnxa.Tensor {
	rows, cols := len(matrix), len(matrix[0])

	// Use pooled memory
	data := e.tensorPool.Get(rows * cols)

	// Fill data (don't defer Put here - caller must handle)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = matrix[i][j]
		}
	}

	return rnxa.NewTensor(data, rows, cols)
}

func (e *enhancedRnxaBackend) releaseTensor(tensor *rnxa.Tensor) {
	if tensor != nil {
		e.tensorPool.Put(tensor.Data())
	}
}

// Enhanced device info with performance metrics
func (e *enhancedRnxaBackend) DeviceInfo() string {
	baseInfo := e.rnxaBackend.DeviceInfo()
	return fmt.Sprintf("%s [Enhanced: GPU threshold=%d, CPU threads=%d, Pool size=%d]",
		baseInfo,
		e.config.MatMulGPUThreshold,
		e.config.MaxCPUThreads,
		e.config.PoolSize)
}

func (e *enhancedRnxaBackend) Close() error {
	// Close batch processing
	if e.batchJobs != nil {
		close(e.batchJobs)
	}

	return e.rnxaBackend.Close()
}

func (e *enhancedRnxaBackend) processBatchVectorOp(vectors [][]float64, operation string) ([][]float64, error) {
	results := make([][]float64, len(vectors)/2)

	for i := 0; i < len(vectors); i += 2 {
		A, B := vectors[i], vectors[i+1]
		var result []float64
		var err error

		switch operation {
		case "add":
			result, err = e.VectorAdd(A, B)
		case "sub":
			result, err = e.VectorSub(A, B)
		case "mul":
			result, err = e.VectorMul(A, B)
		default:
			return nil, fmt.Errorf("unsupported vector operation: %s", operation)
		}

		if err != nil {
			return nil, err
		}
		results[i/2] = result
	}

	return results, nil
}

func (e *enhancedRnxaBackend) ForwardBatch(inputs [][]float64, weights [][]float64, biases []float64, activation string) ([][]float64, error) {
	batchSize := len(inputs)
	config := e.GetPerformanceConfig()

	// Create weight matrix for batch multiplication [input_size × output_size]
	weightMatrix := make([][]float64, len(weights[0]))
	for i := 0; i < len(weights[0]); i++ {
		weightMatrix[i] = make([]float64, len(weights))
		for j := 0; j < len(weights); j++ {
			weightMatrix[i][j] = weights[j][i] // Transpose for correct multiplication
		}
	}

	// Batch matrix multiplication
	var batchOutput [][]float64
	var err error

	if config.ShouldUseGPUForMatMul(batchSize, len(weights[0]), len(weights)) {
		batchOutput, err = e.gpuMatMul(inputs, weightMatrix)
	} else if batchSize*len(weights[0])*len(weights) >= config.MatMulCPUParallel*config.MatMulCPUParallel*config.MatMulCPUParallel {
		batchOutput, err = e.parallelCPUMatMul(inputs, weightMatrix)
	} else {
		batchOutput, err = e.rnxaBackend.MatMul(inputs, weightMatrix)
	}

	if err != nil {
		return nil, fmt.Errorf("batch matrix multiplication failed: %w", err)
	}

	// Add bias and apply activation for each output
	outputs := make([][]float64, batchSize)
	for b := 0; b < batchSize; b++ {
		// Add bias
		biasedOutput := make([]float64, len(biases))
		for j := 0; j < len(biases); j++ {
			biasedOutput[j] = batchOutput[b][j] + biases[j]
		}

		// Apply activation
		if activation == "softmax" {
			outputs[b] = act.SoftmaxVec(biasedOutput)
		} else if config.ShouldUseGPUForActivation(len(biases)) {
			activated, err := e.ActivationFunc(activation, biasedOutput)
			if err == nil {
				outputs[b] = activated
			} else {
				// Fallback to CPU activation
				outputs[b] = make([]float64, len(biases))
				for i, z := range biasedOutput {
					outputs[b][i] = resolveAndApplyActivation(activation, z)
				}
			}
		} else {
			// CPU activation for small outputs
			activated, err := e.ActivationFunc(activation, biasedOutput)
			if err != nil {
				return nil, err
			}
			outputs[b] = activated
		}
	}

	return outputs, nil
}

func (e *enhancedRnxaBackend) GetPerformanceConfig() *PerformanceConfig {
	return e.config
}

func (e *enhancedRnxaBackend) ShouldUseGPUForMatMul(M, N, K int) bool {
	return e.config.ShouldUseGPUForMatMul(M, N, K)
}

func (e *enhancedRnxaBackend) ShouldUseGPUForActivation(size int) bool {
	return e.config.ShouldUseGPUForActivation(size)
}

// Helper function for activation resolution
func resolveAndApplyActivation(activation string, x float64) float64 {
	switch activation {
	case "relu":
		if x > 0 {
			return x
		} else {
			return 0
		}
	case "sigmoid":
		return 1.0 / (1.0 + math.Exp(-x))
	case "tanh":
		return math.Tanh(x)
	case "identity":
		return x
	default:
		return x
	}
}
