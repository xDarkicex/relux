package loss

import (
	"math"

	"github.com/xDarkicex/relux/internal/act"
)

// Categorical Cross-Entropy loss for multi-class classification
type categoricalCrossentropy struct{}

func CategoricalCrossentropy() Loss { return categoricalCrossentropy{} }

func (categoricalCrossentropy) Name() string { return "categorical_crossentropy" }

func (cce categoricalCrossentropy) Forward(yPred, yTrue []float64) float64 {
	const eps = 1e-15
	n := len(yPred)
	if n == 0 {
		return 0
	}

	// Apply softmax to predictions for proper probabilities
	yPredSoftmax := act.SoftmaxVec(yPred)

	var loss float64
	for i := 0; i < n; i++ {
		// Clamp predictions to prevent log(0)
		pred := math.Max(yPredSoftmax[i], eps)
		pred = math.Min(pred, 1-eps)

		// Cross-entropy: -sum(y_true * log(y_pred))
		if yTrue[i] > 0 { // Only add loss for true class (one-hot encoding)
			loss -= yTrue[i] * math.Log(pred)
		}
	}

	return loss
}

func (cce categoricalCrossentropy) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n == 0 {
		return nil
	}

	// For softmax + categorical crossentropy, gradient simplifies to:
	// gradient = y_pred - y_true
	yPredSoftmax := act.SoftmaxVec(yPred)
	grad := make([]float64, n)

	for i := 0; i < n; i++ {
		grad[i] = yPredSoftmax[i] - yTrue[i]
	}

	return grad
}

// Sparse Categorical Cross-Entropy for integer labels
type sparseCategoricalCrossentropy struct{}

func SparseCategoricalCrossentropy() Loss { return sparseCategoricalCrossentropy{} }

func (sparseCategoricalCrossentropy) Name() string { return "sparse_categorical_crossentropy" }

func (scce sparseCategoricalCrossentropy) Forward(yPred, yTrue []float64) float64 {
	const eps = 1e-15
	n := len(yPred)
	if n == 0 || len(yTrue) == 0 {
		return 0
	}

	// Apply softmax to predictions
	yPredSoftmax := act.SoftmaxVec(yPred)

	// yTrue[0] contains the integer class label
	trueClassIdx := int(yTrue[0])
	if trueClassIdx < 0 || trueClassIdx >= n {
		return math.Inf(1) // Invalid class index
	}

	// Clamp prediction for numerical stability
	pred := math.Max(yPredSoftmax[trueClassIdx], eps)
	pred = math.Min(pred, 1-eps)

	return -math.Log(pred)
}

func (scce sparseCategoricalCrossentropy) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n == 0 || len(yTrue) == 0 {
		return make([]float64, n)
	}

	yPredSoftmax := act.SoftmaxVec(yPred)
	grad := make([]float64, n)

	// Copy softmax output
	copy(grad, yPredSoftmax)

	// Subtract 1 from the true class
	trueClassIdx := int(yTrue[0])
	if trueClassIdx >= 0 && trueClassIdx < n {
		grad[trueClassIdx] -= 1.0
	}

	return grad
}
