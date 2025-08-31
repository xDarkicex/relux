package compute

import (
	"fmt"
	"math"

	"github.com/xDarkicex/relux/internal/act"
)

// nativeBackend implements pure Go tensor operations (current relux implementation)
type nativeBackend struct {
	name string
}

func newNativeBackend() ComputeBackend {
	return &nativeBackend{
		name: "Pure Go (Native)",
	}
}

func (n *nativeBackend) MatMul(A, B [][]float64) ([][]float64, error) {
	if len(A) == 0 || len(B) == 0 {
		return nil, fmt.Errorf("empty matrices")
	}
	if len(A[0]) != len(B) {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}

	M, K, N := len(A), len(A[0]), len(B[0])
	C := make([][]float64, M)
	for i := range C {
		C[i] = make([]float64, N)
	}

	// Optimized matrix multiplication with loop reordering
	for i := 0; i < M; i++ {
		for k := 0; k < K; k++ {
			for j := 0; j < N; j++ {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}

	return C, nil
}

func (n *nativeBackend) VectorAdd(A, B []float64) ([]float64, error) {
	if len(A) != len(B) {
		return nil, fmt.Errorf("vector size mismatch: %d != %d", len(A), len(B))
	}

	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] + B[i]
	}
	return result, nil
}

func (n *nativeBackend) VectorSub(A, B []float64) ([]float64, error) {
	if len(A) != len(B) {
		return nil, fmt.Errorf("vector size mismatch: %d != %d", len(A), len(B))
	}

	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] - B[i]
	}
	return result, nil
}

func (n *nativeBackend) VectorMul(A, B []float64) ([]float64, error) {
	if len(A) != len(B) {
		return nil, fmt.Errorf("vector size mismatch: %d != %d", len(A), len(B))
	}

	result := make([]float64, len(A))
	for i := range A {
		result[i] = A[i] * B[i]
	}
	return result, nil
}

func (n *nativeBackend) ActivationFunc(name string, x []float64) ([]float64, error) {
	switch name {
	case "relu":
		result := make([]float64, len(x))
		for i, v := range x {
			if v > 0 {
				result[i] = v
			}
		}
		return result, nil

	case "sigmoid":
		result := make([]float64, len(x))
		for i, v := range x {
			result[i] = 1.0 / (1.0 + math.Exp(-v))
		}
		return result, nil

	case "tanh":
		result := make([]float64, len(x))
		for i, v := range x {
			result[i] = math.Tanh(v)
		}
		return result, nil

	case "softmax":
		return act.SoftmaxVec(x), nil

	default:
		return nil, fmt.Errorf("unsupported activation function: %s", name)
	}
}

func (n *nativeBackend) BatchMatMul(matrices [][][]float64) ([][][]float64, error) {
	if len(matrices)%2 != 0 {
		return nil, fmt.Errorf("batch matmul requires even number of matrices (pairs)")
	}

	results := make([][][]float64, len(matrices)/2)
	for i := 0; i < len(matrices); i += 2 {
		A, B := matrices[i], matrices[i+1]
		result, err := n.MatMul(A, B)
		if err != nil {
			return nil, err
		}
		results[i/2] = result
	}
	return results, nil
}

func (n *nativeBackend) ForwardBatch(inputs [][]float64, weights [][]float64, biases []float64, activation string) ([][]float64, error) {
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
		activated, err := n.ActivationFunc(activation, output)
		if err != nil {
			return nil, err
		}
		outputs[i] = activated
	}

	return outputs, nil
}

func (n *nativeBackend) GetPerformanceConfig() *PerformanceConfig {
	return GetPerformanceConfig()
}

func (n *nativeBackend) ShouldUseGPUForMatMul(M, N, K int) bool {
	return false // Native backend doesn't use GPU
}

func (n *nativeBackend) ShouldUseGPUForActivation(size int) bool {
	return false // Native backend doesn't use GPU
}

func (n *nativeBackend) Name() string       { return n.name }
func (n *nativeBackend) Available() bool    { return true }
func (n *nativeBackend) DeviceInfo() string { return "CPU (Pure Go)" }
func (n *nativeBackend) Close() error       { return nil }
