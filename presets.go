package relux

// SmallMLP creates a configuration for a small multilayer perceptron.
// Suitable for simple problems with small datasets (< 1000 samples).
// Architecture: input -> 32(relu) -> output
func SmallMLP(inputSize, outputSize int) Config {
	return Config{
		Inputs: []InputSpec{{Name: "input", Size: inputSize}},
		Hidden: []LayerSpec{
			{Units: 32, Act: "relu"},
		},
		Output: LayerSpec{Units: outputSize, Act: inferOutputActivation(outputSize)},
		Loss:   inferLoss(outputSize),
	}
}

// MediumMLP creates a configuration for a medium multilayer perceptron.
// Suitable for moderate complexity problems with medium datasets (1K-100K samples).
// Architecture: input -> 128(relu) -> 64(relu) -> output
func MediumMLP(inputSize, outputSize int) Config {
	return Config{
		Inputs: []InputSpec{{Name: "input", Size: inputSize}},
		Hidden: []LayerSpec{
			{Units: 128, Act: "relu"},
			{Units: 64, Act: "relu"},
		},
		Output: LayerSpec{Units: outputSize, Act: inferOutputActivation(outputSize)},
		Loss:   inferLoss(outputSize),
	}
}

// LargeMLP creates a configuration for a large multilayer perceptron.
// Suitable for complex problems with large datasets (> 100K samples).
// Architecture: input -> 512(relu) -> 256(relu) -> 128(relu) -> output
func LargeMLP(inputSize, outputSize int) Config {
	return Config{
		Inputs: []InputSpec{{Name: "input", Size: inputSize}},
		Hidden: []LayerSpec{
			{Units: 512, Act: "relu"},
			{Units: 256, Act: "relu"},
			{Units: 128, Act: "relu"},
		},
		Output: LayerSpec{Units: outputSize, Act: inferOutputActivation(outputSize)},
		Loss:   inferLoss(outputSize),
	}
}

// ClassificationMLP creates a configuration optimized for classification tasks.
// Uses appropriate activations and loss functions for classification.
func ClassificationMLP(inputSize, numClasses int, size string) Config {
	var hidden []LayerSpec
	switch size {
	case "small":
		hidden = []LayerSpec{{Units: 64, Act: "relu"}}
	case "large":
		hidden = []LayerSpec{
			{Units: 256, Act: "relu"},
			{Units: 128, Act: "relu"},
		}
	default: // medium
		hidden = []LayerSpec{
			{Units: 128, Act: "relu"},
			{Units: 64, Act: "relu"},
		}
	}

	// Use proper activations and loss for classification
	outputAct := "sigmoid"
	loss := "bce"
	if numClasses > 2 {
		outputAct = "softmax"             // ✅ NOW USING SOFTMAX!
		loss = "categorical_crossentropy" // ✅ NOW USING PROPER LOSS!
	}

	return Config{
		Inputs: []InputSpec{{Name: "input", Size: inputSize}},
		Hidden: hidden,
		Output: LayerSpec{Units: numClasses, Act: outputAct},
		Loss:   loss,
	}
}

// RegressionMLP creates a configuration optimized for regression tasks.
// Uses appropriate activations and loss functions for regression.
func RegressionMLP(inputSize, outputSize int, size string) Config {
	var hidden []LayerSpec
	switch size {
	case "small":
		hidden = []LayerSpec{{Units: 32, Act: "relu"}}
	case "large":
		hidden = []LayerSpec{
			{Units: 256, Act: "relu"},
			{Units: 128, Act: "relu"},
		}
	default: // medium
		hidden = []LayerSpec{
			{Units: 128, Act: "relu"},
			{Units: 64, Act: "relu"},
		}
	}

	return Config{
		Inputs: []InputSpec{{Name: "input", Size: inputSize}},
		Hidden: hidden,
		Output: LayerSpec{Units: outputSize, Act: "identity"}, // Linear output for regression
		Loss:   "mse",
	}
}

// inferOutputActivation suggests output activation based on output size
func inferOutputActivation(outputSize int) string {
	if outputSize == 1 {
		return "sigmoid" // Binary classification or bounded regression
	}
	return "identity" // Multi-output regression or softmax placeholder
}

// inferLoss suggests loss function based on output size
func inferLoss(outputSize int) string {
	if outputSize == 1 {
		return "bce" // Binary classification
	}
	return "mse" // Regression or multi-output
}
