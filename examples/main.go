package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/xDarkicex/relux"
)

// OptimizationApproach represents a hyperparameter configuration
type OptimizationApproach struct {
	Name             string
	Seed             int64
	HiddenUnits      int
	LearningRate     float64
	Epochs           int
	LRDecay          float64
	EarlyStopping    int
	Momentum         float64
	BatchSize        int
	GradientClip     float64
	ActivationHidden string
	NetworkSize      string
}

// IrisDataset contains the iris classification data
type IrisDataset struct {
	X [][]float64
	Y [][]float64
}

// Result stores the outcome of a training approach
type Result struct {
	Approach OptimizationApproach
	Accuracy float64
	Correct  int
	Total    int
	Error    error
}

func main() {
	fmt.Println("🔍 relux Deep Hyperparameter Search - Iris Classification")
	fmt.Println("=========================================================")

	// Enhanced Iris dataset (24 samples for more robust testing)
	dataset := getEnhancedIrisDataset()

	fmt.Printf("Dataset: %d samples, %d features, %d classes\n",
		len(dataset.X), len(dataset.X[0]), len(dataset.Y[0]))

	// Comprehensive approach matrix - 20 different configurations
	approaches := []OptimizationApproach{
		// Basic approaches
		{"High LR + Small Network", 12345, 64, 0.01, 3000, 0.95, 100, 0.0, 6, 2.0, "relu", "small"},
		{"Balanced + Medium Network", 54321, 128, 0.005, 4000, 0.9, 150, 0.0, 8, 2.0, "relu", "medium"},
		{"Conservative + Large Network", 11111, 256, 0.001, 5000, 0.85, 200, 0.0, 12, 2.0, "relu", "large"},

		// Momentum-based approaches
		{"Momentum + High LR", 22222, 96, 0.02, 2000, 0.8, 50, 0.9, 6, 3.0, "relu", "small"},
		{"Deep + Patient + Momentum", 33333, 192, 0.003, 6000, 0.92, 300, 0.85, 8, 1.5, "relu", "medium"},
		{"Aggressive Momentum", 44444, 80, 0.015, 3500, 0.88, 75, 0.95, 6, 2.5, "relu", "small"},

		// Different activation functions
		{"Tanh Hidden + Conservative", 55555, 128, 0.002, 4500, 0.9, 180, 0.7, 8, 2.0, "tanh", "medium"},
		{"Tanh + High Momentum", 66666, 96, 0.008, 3000, 0.87, 120, 0.9, 6, 2.0, "tanh", "small"},
		{"GELU + Modern Setup", 77777, 144, 0.004, 4000, 0.89, 150, 0.8, 8, 1.8, "gelu", "medium"},
		{"Swish + Mobile Optimized", 88888, 72, 0.012, 2500, 0.85, 80, 0.85, 6, 2.2, "swish", "small"},

		// Learning rate focused
		{"Ultra Low LR + Long Train", 99999, 128, 0.0005, 8000, 0.98, 400, 0.9, 8, 1.0, "relu", "medium"},
		{"Ultra High LR + Fast", 10101, 64, 0.05, 1500, 0.8, 30, 0.95, 4, 5.0, "relu", "small"},
		{"Adaptive + Balanced", 20202, 112, 0.006, 3800, 0.91, 140, 0.75, 8, 2.0, "relu", "medium"},

		// Architectural variations
		{"Wide + Shallow", 30303, 384, 0.003, 4000, 0.9, 160, 0.8, 12, 2.0, "relu", "custom"},
		{"Deep + Narrow", 40404, 48, 0.008, 4500, 0.88, 200, 0.85, 6, 1.5, "relu", "custom"},
		{"Balanced + Patient", 50505, 160, 0.0025, 5500, 0.93, 250, 0.8, 8, 1.8, "relu", "medium"},

		// Edge cases and experimental
		{"Minimal + Aggressive", 60606, 32, 0.025, 2000, 0.75, 40, 0.9, 4, 3.5, "relu", "custom"},
		{"Maximal + Conservative", 70707, 512, 0.0008, 7000, 0.95, 350, 0.7, 16, 1.2, "relu", "custom"},
		{"Random Seed 1", 12121, 88, 0.007, 3200, 0.86, 110, 0.82, 6, 2.1, "relu", "small"},
		{"Random Seed 2", 34343, 176, 0.0035, 4800, 0.91, 190, 0.78, 8, 1.9, "tanh", "medium"},
	}

	// Track results
	var results []Result
	bestAccuracy := 0.0
	bestApproach := ""
	threshold := 80.0 // Target accuracy threshold

	fmt.Printf("\n🎯 Target Accuracy: %.1f%%\n", threshold)
	fmt.Printf("🔄 Testing %d different approaches...\n\n", len(approaches))

	// Iterate through all approaches
	for i, approach := range approaches {
		fmt.Printf("[%d/%d] 🧪 Trying: %s...\n", i+1, len(approaches), approach.Name)

		result := trainAndEvaluate(approach, dataset)
		results = append(results, result)

		if result.Error != nil {
			fmt.Printf("  ❌ Error: %v\n", result.Error)
			continue
		}

		fmt.Printf("  📊 Accuracy: %.1f%% (%d/%d correct)\n",
			result.Accuracy, result.Correct, result.Total)

		// Track best result
		if result.Accuracy > bestAccuracy {
			bestAccuracy = result.Accuracy
			bestApproach = approach.Name
			fmt.Printf("  🎯 New best! (%.1f%%)\n", bestAccuracy)

			// Early success check
			if bestAccuracy >= threshold {
				fmt.Printf("  🎉 Target accuracy reached!\n")
			}
		}

		// Add small delay for readability
		time.Sleep(100 * time.Millisecond)
	}

	// Final results summary
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🏆 FINAL RESULTS")
	fmt.Println(strings.Repeat("=", 60))

	if bestAccuracy >= threshold {
		fmt.Printf("✅ SUCCESS: Target accuracy reached!\n")
		fmt.Printf("🥇 Best approach: %s\n", bestApproach)
		fmt.Printf("🎯 Best accuracy: %.1f%%\n", bestAccuracy)
	} else {
		fmt.Printf("⚠️  Best result: %s with %.1f%% accuracy (below %.1f%% threshold)\n",
			bestApproach, bestAccuracy, threshold)
		fmt.Printf("💡 Tip: Dataset might need preprocessing or more sophisticated architecture.\n")
	}

	// Show top 5 results
	fmt.Println("\n📈 Top 5 Approaches:")
	topResults := getTopResults(results, 5)
	for i, result := range topResults {
		status := "❌"
		if result.Accuracy >= threshold {
			status = "✅"
		}
		fmt.Printf("  %d. %s: %.1f%% %s\n",
			i+1, result.Approach.Name, result.Accuracy, status)
	}

	// Additional analysis
	analyzeResults(results, threshold)
}

func trainAndEvaluate(approach OptimizationApproach, dataset IrisDataset) Result {
	// Create network configuration
	var config relux.Config

	switch approach.NetworkSize {
	case "custom":
		config = relux.Config{
			Inputs: []relux.InputSpec{{Name: "input", Size: 4}},
			Hidden: []relux.LayerSpec{{Units: approach.HiddenUnits, Act: approach.ActivationHidden}},
			Output: relux.LayerSpec{Units: 3, Act: "softmax"},
			Loss:   "categorical_crossentropy",
		}
	default:
		config = relux.ClassificationMLP(4, 3, approach.NetworkSize)
		// Override activation if specified
		if approach.ActivationHidden != "relu" {
			for i := range config.Hidden {
				config.Hidden[i].Act = approach.ActivationHidden
			}
		}
		// Override units if specified
		if approach.HiddenUnits != 0 {
			config.Hidden[0].Units = approach.HiddenUnits
		}
	}

	// Create network
	net, err := relux.NewNetwork(
		relux.WithConfig(config),
		relux.WithSeed(approach.Seed),
		relux.WithAcceleration("auto"),
	)
	if err != nil {
		return Result{Approach: approach, Error: err}
	}

	// Train network
	trainOptions := []relux.TrainOption{
		relux.Epochs(approach.Epochs),
		relux.LearningRate(approach.LearningRate),
		relux.LearningRateDecay(approach.LRDecay, 500),
		relux.EarlyStopping(approach.EarlyStopping),
		relux.BatchSize(approach.BatchSize),
		relux.GradientClip(approach.GradientClip),
		relux.Verbose(false),
		relux.Shuffle(true),
	}

	// Add momentum if specified
	if approach.Momentum > 0 {
		trainOptions = append(trainOptions, relux.Momentum(approach.Momentum))
	}

	err = net.Fit(dataset.X, dataset.Y, trainOptions...)
	if err != nil {
		return Result{Approach: approach, Error: err}
	}

	// Evaluate accuracy
	correct := 0
	total := len(dataset.X)

	for i, x := range dataset.X {
		pred, err := net.Predict(x)
		if err != nil {
			return Result{Approach: approach, Error: err}
		}

		// Find predicted class
		maxIdx := 0
		for j := 1; j < len(pred); j++ {
			if pred[j] > pred[maxIdx] {
				maxIdx = j
			}
		}

		// Find true class
		trueIdx := 0
		for j := 1; j < len(dataset.Y[i]); j++ {
			if dataset.Y[i][j] > dataset.Y[i][trueIdx] {
				trueIdx = j
			}
		}

		if maxIdx == trueIdx {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total) * 100

	return Result{
		Approach: approach,
		Accuracy: accuracy,
		Correct:  correct,
		Total:    total,
		Error:    nil,
	}
}

func getEnhancedIrisDataset() IrisDataset {
	// Enhanced iris dataset with 24 samples (8 per class)
	X := [][]float64{
		// Setosa (8 samples)
		{5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2}, {4.6, 3.1, 1.5, 0.2},
		{5.0, 3.6, 1.4, 0.2}, {5.4, 3.9, 1.7, 0.4}, {4.6, 3.4, 1.4, 0.3}, {5.0, 3.4, 1.5, 0.2},

		// Versicolor (8 samples)
		{7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5}, {5.5, 2.3, 4.0, 1.3},
		{6.5, 2.8, 4.6, 1.5}, {5.7, 2.8, 4.5, 1.3}, {6.3, 3.3, 4.7, 1.6}, {4.9, 2.4, 3.3, 1.0},

		// Virginica (8 samples)
		{6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1}, {6.3, 2.9, 5.6, 1.8},
		{6.5, 3.0, 5.8, 2.2}, {7.6, 3.0, 6.6, 2.1}, {4.9, 2.5, 4.5, 1.7}, {7.3, 2.9, 6.3, 1.8},
	}

	Y := [][]float64{
		// Setosa (8 samples)
		{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},

		// Versicolor (8 samples)
		{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},

		// Virginica (8 samples)
		{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
	}

	return IrisDataset{X: X, Y: Y}
}

func getTopResults(results []Result, top int) []Result {
	// Simple bubble sort by accuracy (descending)
	sorted := make([]Result, len(results))
	copy(sorted, results)

	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j].Accuracy < sorted[j+1].Accuracy {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	if top > len(sorted) {
		top = len(sorted)
	}

	return sorted[:top]
}

func analyzeResults(results []Result, threshold float64) {
	if len(results) == 0 {
		return
	}

	fmt.Println("\n📊 Analysis:")

	// Success rate
	successful := 0
	totalValid := 0
	var accuracies []float64

	for _, result := range results {
		if result.Error == nil {
			totalValid++
			accuracies = append(accuracies, result.Accuracy)
			if result.Accuracy >= threshold {
				successful++
			}
		}
	}

	if totalValid == 0 {
		fmt.Println("  ❌ No valid results to analyze")
		return
	}

	successRate := float64(successful) / float64(totalValid) * 100
	fmt.Printf("  Success rate: %.1f%% (%d/%d approaches reached target)\n",
		successRate, successful, totalValid)

	// Average accuracy
	sum := 0.0
	for _, acc := range accuracies {
		sum += acc
	}
	avgAccuracy := sum / float64(len(accuracies))
	fmt.Printf("  Average accuracy: %.1f%%\n", avgAccuracy)

	// Find best activation function
	activationStats := make(map[string][]float64)
	for _, result := range results {
		if result.Error == nil {
			act := result.Approach.ActivationHidden
			activationStats[act] = append(activationStats[act], result.Accuracy)
		}
	}

	fmt.Println("  Best activation functions:")
	for act, accs := range activationStats {
		if len(accs) > 0 {
			sum := 0.0
			for _, acc := range accs {
				sum += acc
			}
			avg := sum / float64(len(accs))
			fmt.Printf("    %s: %.1f%% average (%d trials)\n", act, avg, len(accs))
		}
	}

	// Recommendations
	fmt.Println("\n💡 Recommendations:")
	if successRate < 25 {
		fmt.Println("  • Consider data preprocessing (normalization, feature scaling)")
		fmt.Println("  • Try different network architectures (deeper networks)")
		fmt.Println("  • Dataset might be too challenging for current setup")
	} else if successRate < 50 {
		fmt.Println("  • Some approaches work - focus on successful patterns")
		fmt.Println("  • Consider ensemble methods")
	} else {
		fmt.Println("  • Multiple approaches successful - framework is well-tuned")
		fmt.Println("  • Consider production deployment")
	}
}
