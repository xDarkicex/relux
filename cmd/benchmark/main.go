package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/xDarkicex/relux"
	"github.com/xDarkicex/relux/internal/compute"
	"github.com/xDarkicex/relux/internal/layer"
)

func main() {
	var (
		matmul     = flag.Bool("matmul", true, "Run matrix multiplication benchmarks")
		activation = flag.Bool("activation", true, "Run activation function benchmarks")
		network    = flag.Bool("network", true, "Run neural network benchmarks")
		verbose    = flag.Bool("v", false, "Verbose output")
	)
	flag.Parse()

	fmt.Println("🚀 relux Phase 2 Performance Benchmarks")
	fmt.Println("========================================")

	// Show available backends
	backends := compute.GetAvailableBackends()
	fmt.Printf("Available backends: %v\n", backends)

	// Show performance configuration
	config := compute.GetPerformanceConfig()
	fmt.Printf("GPU thresholds: MatMul=%dx%d, Vector=%d, Activation=%d\n",
		config.MatMulGPUThreshold, config.MatMulGPUThreshold,
		config.VectorGPUThreshold, config.ActivationGPUThreshold)

	// Run benchmarks
	suite := compute.NewBenchmarkSuite()

	if *matmul {
		fmt.Println("\n⚡ Running matrix multiplication benchmarks...")
		suite.RunMatMulBenchmarks()
	}

	if *activation {
		fmt.Println("\n🎯 Running activation function benchmarks...")
		suite.RunActivationBenchmarks()
	}

	if *network {
		fmt.Println("\n🧠 Running neural network benchmarks...")
		suite.RunNetworkBenchmarks()
	}

	// Display results
	suite.PrintResults()

	// Test actual relux network performance
	fmt.Println("\n🔬 Testing real relux network performance...")
	testReluxPerformance(*verbose)
}

func testReluxPerformance(verbose bool) {
	// Create different sized networks
	configs := []struct {
		name   string
		config relux.Config
	}{
		{
			"Small Network",
			relux.SmallMLP(784, 10),
		},
		{
			"Medium Network",
			relux.MediumMLP(1024, 50),
		},
		{
			"Large Network",
			relux.LargeMLP(2048, 100),
		},
	}

	for _, testConfig := range configs {
		fmt.Printf("\n📊 %s Performance:\n", testConfig.name)

		// Test with different acceleration settings
		accelerationModes := []string{"auto", "native"}

		for _, mode := range accelerationModes {
			net, err := relux.NewNetwork(
				relux.WithConfig(testConfig.config),
				relux.WithAcceleration(mode),
			)
			if err != nil {
				log.Printf("Failed to create %s network with %s: %v", testConfig.name, mode, err)
				continue
			}

			// Get performance info if available
			if len(net.GetLayers()) > 0 {
				if dense, ok := net.GetLayers()[0].(*layer.Dense); ok {
					if perfInfo := dense.GetPerformanceInfo(); verbose {
						fmt.Printf("  %s mode: %v\n", mode, perfInfo)
					}
				}
			}

			// Test prediction performance
			testInput := make([]float64, testConfig.config.Inputs[0].Size)
			for i := range testInput {
				testInput[i] = 0.5 // Dummy data
			}

			// Warmup
			for i := 0; i < 3; i++ {
				net.Predict(testInput)
			}

			// Benchmark
			start := time.Now()
			iterations := 100
			for i := 0; i < iterations; i++ {
				net.Predict(testInput)
			}
			duration := time.Since(start) / time.Duration(iterations)

			fmt.Printf("  %s: %s per prediction\n", mode, duration)
		}
	}
}
