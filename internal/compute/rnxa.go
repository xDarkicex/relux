//go:build rnxa
// +build rnxa

package compute

import (
	"context"
	"fmt"

	"github.com/xDarkicex/rnxa"
)

// rnxaBackend implements hardware-accelerated tensor operations via rnxa
type rnxaBackend struct {
	engine rnxa.ComputeEngine
	ctx    context.Context
	name   string
}

func newRnxaBackend() (ComputeBackend, error) {
	engine, err := rnxa.NewEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create rnxa engine: %w", err)
	}

	device := engine.Device()
	name := fmt.Sprintf("rnxa (%s: %s)", device.Platform, device.Name)

	return &rnxaBackend{
		engine: engine,
		ctx:    context.Background(),
		name:   name,
	}, nil
}

func (r *rnxaBackend) MatMul(A, B [][]float64) ([][]float64, error) {
	if len(A) == 0 || len(B) == 0 {
		return nil, fmt.Errorf("empty matrices")
	}
	if len(A[0]) != len(B) {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}

	// Convert [][]float64 to rnxa tensors
	tensorA := convertMatrixToTensor(A)
	tensorB := convertMatrixToTensor(B)

	// Hardware-accelerated matrix multiplication
	result, err := r.engine.MatMul(r.ctx, tensorA, tensorB)
	if err != nil {
		return nil, fmt.Errorf("rnxa MatMul failed: %w", err)
	}

	// Convert back to [][]float64
	return convertTensorToMatrix(result), nil
}

func (r *rnxaBackend) VectorAdd(A, B []float64) ([]float64, error) {
	// For small vectors, use CPU (no transfer overhead)
	if len(A) < 1000 {
		return nativeVectorAdd(A, B)
	}

	tensorA := rnxa.NewTensor(A)
	tensorB := rnxa.NewTensor(B)

	result, err := r.engine.VectorAdd(r.ctx, tensorA, tensorB)
	if err != nil {
		return nil, fmt.Errorf("rnxa VectorAdd failed: %w", err)
	}

	return result.Data(), nil
}

func (r *rnxaBackend) VectorSub(A, B []float64) ([]float64, error) {
	if len(A) < 1000 {
		return nativeVectorSub(A, B)
	}

	tensorA := rnxa.NewTensor(A)
	tensorB := rnxa.NewTensor(B)

	result, err := r.engine.VectorSub(r.ctx, tensorA, tensorB)
	if err != nil {
		return nil, fmt.Errorf("rnxa VectorSub failed: %w", err)
	}

	return result.Data(), nil
}

func (r *rnxaBackend) VectorMul(A, B []float64) ([]float64, error) {
	if len(A) < 1000 {
		return nativeVectorMul(A, B)
	}

	tensorA := rnxa.NewTensor(A)
	tensorB := rnxa.NewTensor(B)

	result, err := r.engine.VectorMul(r.ctx, tensorA, tensorB)
	if err != nil {
		return nil, fmt.Errorf("rnxa VectorMul failed: %w", err)
	}

	return result.Data(), nil
}

func (r *rnxaBackend) ActivationFunc(name string, x []float64) ([]float64, error) {
	// Small vectors: use CPU
	if len(x) < 500 {
		return nativeActivation(name, x)
	}

	tensor := rnxa.NewTensor(x)
	var result *rnxa.Tensor
	var err error

	switch name {
	case "relu":
		result, err = r.engine.ReLU(r.ctx, tensor)
	case "sigmoid":
		result, err = r.engine.Sigmoid(r.ctx, tensor)
	case "tanh":
		result, err = r.engine.Tanh(r.ctx, tensor)
	case "softmax":
		result, err = r.engine.Softmax(r.ctx, tensor)
	default:
		return nil, fmt.Errorf("unsupported activation function: %s", name)
	}

	if err != nil {
		return nil, fmt.Errorf("rnxa %s failed: %w", name, err)
	}

	return result.Data(), nil
}

func (r *rnxaBackend) BatchMatMul(matrices [][][]float64) ([][][]float64, error) {
	if len(matrices)%2 != 0 {
		return nil, fmt.Errorf("batch matmul requires even number of matrices (pairs)")
	}

	results := make([][][]float64, len(matrices)/2)
	for i := 0; i < len(matrices); i += 2 {
		A, B := matrices[i], matrices[i+1]
		result, err := r.MatMul(A, B)
		if err != nil {
			return nil, err
		}
		results[i/2] = result
	}
	return results, nil
}

func (r *rnxaBackend) ForwardBatch(inputs [][]float64, weights [][]float64, biases []float64, activation string) ([][]float64, error) {
	batchSize := len(inputs)
	outputs := make([][]float64, batchSize)

	for i, input := range inputs {
		// Standard forward pass: input * weights^T + bias
		output := make([]float64, len(biases))

		for j := 0; j < len(weights); j++ {
			sum := biases[j]
			for k := 0; k < len(input); k++ {
				sum += input[k] * weights[j][k]
			}
			output[j] = sum
		}

		// Apply activation
		activated, err := r.ActivationFunc(activation, output)
		if err != nil {
			return nil, err
		}
		outputs[i] = activated
	}

	return outputs, nil
}

func (r *rnxaBackend) GetPerformanceConfig() *PerformanceConfig {
	return GetPerformanceConfig()
}

func (r *rnxaBackend) ShouldUseGPUForMatMul(M, N, K int) bool {
	config := GetPerformanceConfig()
	return config.ShouldUseGPUForMatMul(M, N, K)
}

func (r *rnxaBackend) ShouldUseGPUForActivation(size int) bool {
	config := GetPerformanceConfig()
	return config.ShouldUseGPUForActivation(size)
}

func (r *rnxaBackend) Name() string    { return r.name }
func (r *rnxaBackend) Available() bool { return r.engine.Available() }
func (r *rnxaBackend) DeviceInfo() string {
	device := r.engine.Device()
	memory := r.engine.Memory()
	return fmt.Sprintf("%s (%d cores, %.1fGB memory)",
		device.Name, device.Cores, float64(memory.Available)/1e9)
}
func (r *rnxaBackend) Close() error { return r.engine.Close() }

// Helper functions for tensor conversion
func convertMatrixToTensor(matrix [][]float64) *rnxa.Tensor {
	rows, cols := len(matrix), len(matrix[0])
	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = matrix[i][j]
		}
	}

	return rnxa.NewTensor(data, rows, cols)
}

func convertTensorToMatrix(tensor *rnxa.Tensor) [][]float64 {
	shape := tensor.Shape()
	rows, cols := shape[0], shape[1]
	data := tensor.Data()

	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = data[i*cols+j]
		}
	}

	return matrix
}

// Native fallbacks for small operations
func nativeVectorAdd(A, B []float64) ([]float64, error) {
	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] + B[i]
	}
	return result, nil
}

func nativeVectorSub(A, B []float64) ([]float64, error) {
	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] - B[i]
	}
	return result, nil
}

func nativeVectorMul(A, B []float64) ([]float64, error) {
	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] * B[i]
	}
	return result, nil
}

func nativeActivation(name string, x []float64) ([]float64, error) {
	// Reuse native backend implementation
	native := newNativeBackend()
	return native.ActivationFunc(name, x)
}
