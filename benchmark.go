package relux

import (
	"fmt"
	"time"

	"github.com/xDarkicex/relux/internal/layer"
)

// BenchmarkResult contains performance metrics for a network operation
type BenchmarkResult struct {
	Operation   string        // "Predict", "Training", "Batch"
	Duration    time.Duration // Average time per operation
	Throughput  float64       // Operations per second
	NetworkInfo string        // Network architecture info
	BackendInfo string        // Backend being used
}

// Benchmark runs performance tests on the network and returns results
func (n *Network) Benchmark() BenchmarkResult {
	if n == nil || len(n.layers) == 0 {
		return BenchmarkResult{
			Operation: "Error",
			Duration:  0,
		}
	}

	// Create dummy input
	dummyInput := make([]float64, n.inputSize)
	for i := range dummyInput {
		dummyInput[i] = 0.5
	}

	// Warmup
	for i := 0; i < 5; i++ {
		n.Predict(dummyInput)
	}

	// Benchmark prediction
	iterations := 100
	start := time.Now()
	for i := 0; i < iterations; i++ {
		n.Predict(dummyInput)
	}
	avgDuration := time.Since(start) / time.Duration(iterations)

	// Get backend info if available
	backendInfo := "Native Go"
	if len(n.layers) > 0 {
		if dense, ok := n.layers[0].(*layer.Dense); ok {
			if perfInfo := dense.GetPerformanceInfo(); perfInfo != nil {
				if backend, ok := perfInfo["backend"].(string); ok {
					backendInfo = backend
				}
			}
		}
	}

	return BenchmarkResult{
		Operation:   "Predict",
		Duration:    avgDuration,
		Throughput:  1.0 / avgDuration.Seconds(),
		NetworkInfo: n.Architecture(),
		BackendInfo: backendInfo,
	}
}

// BenchmarkBatch tests batch prediction performance
func (n *Network) BenchmarkBatch(batchSize int) BenchmarkResult {
	if batchSize <= 0 {
		batchSize = 32
	}

	// Create dummy batch
	batch := make([][]float64, batchSize)
	for i := range batch {
		batch[i] = make([]float64, n.inputSize)
		for j := range batch[i] {
			batch[i][j] = 0.5
		}
	}

	// Warmup
	n.PredictBatch(batch)

	// Benchmark
	iterations := 20
	start := time.Now()
	for i := 0; i < iterations; i++ {
		n.PredictBatch(batch)
	}
	avgDuration := time.Since(start) / time.Duration(iterations)

	return BenchmarkResult{
		Operation:   fmt.Sprintf("Batch[%d]", batchSize),
		Duration:    avgDuration,
		Throughput:  float64(batchSize) / avgDuration.Seconds(),
		NetworkInfo: n.Architecture(),
		BackendInfo: n.getBackendInfo(),
	}
}

// BenchmarkTraining tests training performance on dummy data
func (n *Network) BenchmarkTraining(epochs int) BenchmarkResult {
	if epochs <= 0 {
		epochs = 100
	}

	// Create dummy training data
	X := make([][]float64, 10)
	Y := make([][]float64, 10)
	outputSize := n.layers[len(n.layers)-1].(*layer.Dense).OutputSize()

	for i := range X {
		X[i] = make([]float64, n.inputSize)
		Y[i] = make([]float64, outputSize)
		for j := range X[i] {
			X[i][j] = 0.5
		}
		for j := range Y[i] {
			Y[i][j] = 0.5
		}
	}

	// Benchmark training
	start := time.Now()
	n.Fit(X, Y,
		Epochs(epochs),
		LearningRate(0.01),
		Verbose(false),
	)
	duration := time.Since(start)

	return BenchmarkResult{
		Operation:   fmt.Sprintf("Training[%d epochs]", epochs),
		Duration:    duration,
		Throughput:  float64(epochs) / duration.Seconds(),
		NetworkInfo: n.Architecture(),
		BackendInfo: n.getBackendInfo(),
	}
}

// Helper method to get backend info
func (n *Network) getBackendInfo() string {
	if len(n.layers) > 0 {
		if dense, ok := n.layers[0].(*layer.Dense); ok {
			if perfInfo := dense.GetPerformanceInfo(); perfInfo != nil {
				if backend, ok := perfInfo["backend"].(string); ok {
					return backend
				}
			}
		}
	}
	return "Native Go"
}

// String returns a formatted benchmark result
func (br BenchmarkResult) String() string {
	return fmt.Sprintf("%s: %v (%.1f/sec) on %s using %s",
		br.Operation, br.Duration, br.Throughput, br.NetworkInfo, br.BackendInfo)
}

// Compare compares two networks' performance
func CompareBenchmarks(net1, net2 *Network) (BenchmarkResult, BenchmarkResult, float64) {
	result1 := net1.Benchmark()
	result2 := net2.Benchmark()

	speedup := float64(result1.Duration) / float64(result2.Duration)
	return result1, result2, speedup
}
