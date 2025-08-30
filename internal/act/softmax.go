package act

import (
	"math"
)

// Softmax activation for multi-class classification
type softmax struct{}

func Softmax() Activation { return softmax{} }

func (softmax) Name() string { return "softmax" }

func (s softmax) Apply(x float64) float64 {
	// Softmax is applied to vectors, not scalars
	// This single-value version isn't mathematically correct
	// but needed for interface compatibility
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s softmax) Derivative(x float64) float64 {
	// For interface compatibility - actual derivative computed in ApplyVec
	sig := s.Apply(x)
	return sig * (1 - sig)
}

// SoftmaxVec applies softmax to a complete vector (the correct way)
func SoftmaxVec(z []float64) []float64 {
	if len(z) == 0 {
		return []float64{}
	}

	// Find max for numerical stability
	maxVal := z[0]
	for _, val := range z[1:] {
		if val > maxVal {
			maxVal = val
		}
	}

	// Compute exp(z_i - max) and sum
	exp := make([]float64, len(z))
	var sum float64
	for i, val := range z {
		exp[i] = math.Exp(val - maxVal)
		sum += exp[i]
	}

	// Normalize
	out := make([]float64, len(z))
	for i, val := range exp {
		out[i] = val / sum
	}
	return out
}

// SoftmaxDerivativeVec computes the Jacobian matrix for softmax
func SoftmaxDerivativeVec(z []float64) [][]float64 {
	softmaxOut := SoftmaxVec(z)
	n := len(z)
	jacobian := make([][]float64, n)

	for i := range jacobian {
		jacobian[i] = make([]float64, n)
		for j := range jacobian[i] {
			if i == j {
				jacobian[i][j] = softmaxOut[i] * (1 - softmaxOut[i])
			} else {
				jacobian[i][j] = -softmaxOut[i] * softmaxOut[j]
			}
		}
	}
	return jacobian
}
