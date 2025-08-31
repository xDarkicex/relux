package compute

import (
	"fmt"
	"testing"
)

func TestBackendSelection(t *testing.T) {
	// Test automatic backend selection
	backend := NewComputeBackend()
	defer backend.Close()

	if !backend.Available() {
		t.Error("Backend should be available")
	}

	t.Logf("Selected backend: %s", backend.Name())
	t.Logf("Device info: %s", backend.DeviceInfo())
}

func TestMatrixMultiplication(t *testing.T) {
	backend := NewComputeBackend()
	defer backend.Close()

	// Test small matrix multiplication
	A := [][]float64{{1, 2}, {3, 4}}
	B := [][]float64{{5, 6}, {7, 8}}

	result, err := backend.MatMul(A, B)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := [][]float64{{19, 22}, {43, 50}}

	for i := range result {
		for j := range result[i] {
			if result[i][j] != expected[i][j] {
				t.Errorf("Result[%d][%d] = %f, expected %f",
					i, j, result[i][j], expected[i][j])
			}
		}
	}

	t.Logf("MatMul test passed with backend: %s", backend.Name())
}

func TestActivationFunctions(t *testing.T) {
	backend := NewComputeBackend()
	defer backend.Close()

	input := []float64{-1, 0, 1, 2}

	activations := []string{"relu", "sigmoid", "tanh"}

	for _, activation := range activations {
		result, err := backend.ActivationFunc(activation, input)
		if err != nil {
			t.Errorf("Activation %s failed: %v", activation, err)
			continue
		}

		if len(result) != len(input) {
			t.Errorf("Activation %s result length mismatch", activation)
			continue
		}

		t.Logf("Activation %s: %v -> %v", activation, input, result)
	}
}

func BenchmarkMatMul(b *testing.B) {
	backend := NewComputeBackend()
	defer backend.Close()

	// Benchmark with different sizes
	sizes := []int{32, 128, 256}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			A := make([][]float64, size)
			B := make([][]float64, size)
			for i := range A {
				A[i] = make([]float64, size)
				B[i] = make([]float64, size)
				for j := range A[i] {
					A[i][j] = float64(i + j)
					B[i][j] = float64(i * j)
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				backend.MatMul(A, B)
			}
		})
	}
}
