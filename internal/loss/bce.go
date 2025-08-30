package loss

import "math"

// Forward computes binary cross-entropy
func (bce) Forward(yPred, yTrue []float64) float64 {
	const eps = 1e-15
	n := len(yPred)
	if n == 0 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		p := clamp(yPred[i], eps, 1-eps)
		y := clamp(yTrue[i], 0, 1)
		s += -(y*math.Log(p) + (1-y)*math.Log(1-p))
	}
	return s / float64(n)
}

// Backward computes gradient of BCE w.r.t. predictions
func (bce) Backward(yPred, yTrue []float64) []float64 {
	const eps = 1e-15
	n := len(yPred)
	if n == 0 {
		return nil
	}
	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		p := clamp(yPred[i], eps, 1-eps)
		y := clamp(yTrue[i], 0, 1)
		grad[i] = (p - y) / (p * (1 - p) * float64(n))
	}
	return grad
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
