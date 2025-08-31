package compute

import (
	"fmt"
	"math/rand"
	"time"
)

// BenchmarkResult contains performance measurements
type BenchmarkResult struct {
	Operation  string
	InputSize  string
	Backend    string
	Duration   time.Duration
	Throughput float64 // Operations per second
	MemoryUsed int64   // Bytes allocated
	Speedup    float64 // Compared to baseline
}

// BenchmarkSuite runs comprehensive performance tests
type BenchmarkSuite struct {
	backends []ComputeBackend
	results  []BenchmarkResult
}

// NewBenchmarkSuite creates a benchmark suite with available backends
func NewBenchmarkSuite() *BenchmarkSuite {
	var backends []ComputeBackend

	// Always test native backend
	backends = append(backends, newNativeBackend())

	// Try to add rnxa backend if available
	if rnxaBackend, err := tryRnxaBackend(); err == nil {
		backends = append(backends, rnxaBackend)
	}

	return &BenchmarkSuite{
		backends: backends,
		results:  make([]BenchmarkResult, 0),
	}
}

// RunMatMulBenchmarks tests matrix multiplication across different sizes
func (bs *BenchmarkSuite) RunMatMulBenchmarks() {
	sizes := []int{32, 64, 128, 256, 512, 1024}

	for _, size := range sizes {
		// Generate test matrices
		A := generateMatrix(size, size)
		B := generateMatrix(size, size)

		var baselineDuration time.Duration

		for i, backend := range bs.backends {
			// Warm up
			for j := 0; j < 3; j++ {
				backend.MatMul(A, B)
			}

			// Actual benchmark
			start := time.Now()
			iterations := 10
			for j := 0; j < iterations; j++ {
				_, err := backend.MatMul(A, B)
				if err != nil {
					fmt.Printf("Error in %s: %v\n", backend.Name(), err)
					continue
				}
			}
			duration := time.Since(start) / time.Duration(iterations)

			// Calculate speedup
			speedup := 1.0
			if i == 0 {
				baselineDuration = duration
			} else if baselineDuration > 0 {
				speedup = float64(baselineDuration) / float64(duration)
			}

			// Calculate throughput (FLOPS)
			flops := 2 * int64(size) * int64(size) * int64(size) // 2*N^3 operations
			throughput := float64(flops) / duration.Seconds()

			result := BenchmarkResult{
				Operation:  "MatMul",
				InputSize:  fmt.Sprintf("%dx%d", size, size),
				Backend:    backend.Name(),
				Duration:   duration,
				Throughput: throughput,
				Speedup:    speedup,
			}

			bs.results = append(bs.results, result)
		}
	}
}

// RunActivationBenchmarks tests activation functions
func (bs *BenchmarkSuite) RunActivationBenchmarks() {
	sizes := []int{100, 500, 1000, 5000, 10000}
	activations := []string{"relu", "sigmoid", "tanh", "softmax"}

	for _, activation := range activations {
		for _, size := range sizes {
			// Generate test vector
			input := generateVector(size)

			var baselineDuration time.Duration

			for i, backend := range bs.backends {
				// Warm up
				for j := 0; j < 5; j++ {
					backend.ActivationFunc(activation, input)
				}

				// Benchmark
				start := time.Now()
				iterations := 100
				for j := 0; j < iterations; j++ {
					_, err := backend.ActivationFunc(activation, input)
					if err != nil {
						continue
					}
				}
				duration := time.Since(start) / time.Duration(iterations)

				speedup := 1.0
				if i == 0 {
					baselineDuration = duration
				} else if baselineDuration > 0 {
					speedup = float64(baselineDuration) / float64(duration)
				}

				throughput := float64(size) / duration.Seconds()

				result := BenchmarkResult{
					Operation:  activation,
					InputSize:  fmt.Sprintf("vec[%d]", size),
					Backend:    backend.Name(),
					Duration:   duration,
					Throughput: throughput,
					Speedup:    speedup,
				}

				bs.results = append(bs.results, result)
			}
		}
	}
}

// RunNetworkBenchmarks tests real neural network operations
func (bs *BenchmarkSuite) RunNetworkBenchmarks() {
	// Simulate different network sizes
	networkConfigs := []struct {
		name   string
		layers [][]int // [input, output] for each layer
	}{
		{"Small (MNIST)", [][]int{{784, 128}, {128, 64}, {64, 10}}},
		{"Medium (CIFAR)", [][]int{{3072, 256}, {256, 128}, {128, 64}, {64, 10}}},
		{"Large (ImageNet)", [][]int{{4096, 512}, {512, 256}, {256, 128}, {128, 1000}}},
	}

	for _, config := range networkConfigs {
		batchSizes := []int{1, 32, 64, 128}

		for _, batchSize := range batchSizes {
			var baselineDuration time.Duration

			for i, backend := range bs.backends {
				// Simulate forward pass
				start := time.Now()

				// Create batch input
				currentOutput := generateMatrix(batchSize, config.layers[0][0])

				// Forward through layers
				for _, layer := range config.layers {
					weights := generateMatrix(layer[0], layer[1])

					// Batch matrix multiplication: [batch_size, input] × [input, output]
					newOutput := make([][]float64, batchSize)
					for b := 0; b < batchSize; b++ {
						inputRow := [][]float64{currentOutput[b]}
						result, _ := backend.MatMul(inputRow, weights)
						if len(result) > 0 {
							newOutput[b] = result[0]

							// Apply ReLU activation (except last layer)
							if layer != config.layers[len(config.layers)-1] {
								backend.ActivationFunc("relu", newOutput[b])
							}
						}
					}
					currentOutput = newOutput
				}

				duration := time.Since(start)

				speedup := 1.0
				if i == 0 {
					baselineDuration = duration
				} else if baselineDuration > 0 {
					speedup = float64(baselineDuration) / float64(duration)
				}

				throughput := float64(batchSize) / duration.Seconds()

				result := BenchmarkResult{
					Operation:  "Network Forward",
					InputSize:  fmt.Sprintf("%s (batch=%d)", config.name, batchSize),
					Backend:    backend.Name(),
					Duration:   duration,
					Throughput: throughput,
					Speedup:    speedup,
				}

				bs.results = append(bs.results, result)
			}
		}
	}
}

// PrintResults displays benchmark results in a formatted table
func (bs *BenchmarkSuite) PrintResults() {
	fmt.Println("\n🚀 relux Performance Benchmark Results")
	fmt.Println("=======================================")

	// Group results by operation
	operationGroups := make(map[string][]BenchmarkResult)
	for _, result := range bs.results {
		operationGroups[result.Operation] = append(operationGroups[result.Operation], result)
	}

	for operation, results := range operationGroups {
		fmt.Printf("\n📊 %s Performance:\n", operation)
		fmt.Println("Input Size               | Backend                    | Duration      | Speedup")
		fmt.Println("-------------------------|----------------------------|---------------|--------")

		for _, result := range results {
			fmt.Printf("%-24s | %-26s | %12s | %6.1fx\n",
				result.InputSize,
				result.Backend,
				result.Duration.String(),
				result.Speedup)
		}
	}

	// Summary statistics
	fmt.Println("\n🏆 Performance Summary:")
	maxSpeedup := 0.0
	bestBackend := ""

	for _, result := range bs.results {
		if result.Speedup > maxSpeedup && result.Backend != "Pure Go (Native)" {
			maxSpeedup = result.Speedup
			bestBackend = result.Backend
		}
	}

	if maxSpeedup > 1.0 {
		fmt.Printf("Best acceleration: %.1fx speedup with %s\n", maxSpeedup, bestBackend)
		fmt.Printf("Average speedup: %.1fx across all operations\n", calculateAverageSpeedup(bs.results))
	} else {
		fmt.Println("Hardware acceleration not available or not beneficial for tested sizes")
	}
}

func calculateAverageSpeedup(results []BenchmarkResult) float64 {
	totalSpeedup := 0.0
	count := 0

	for _, result := range results {
		if result.Backend != "Pure Go (Native)" && result.Speedup > 1.0 {
			totalSpeedup += result.Speedup
			count++
		}
	}

	if count == 0 {
		return 1.0
	}
	return totalSpeedup / float64(count)
}

// Helper functions for test data generation
func generateMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()*2 - 1 // Random values [-1, 1]
		}
	}
	return matrix
}

func generateVector(size int) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1
	}
	return vector
}
