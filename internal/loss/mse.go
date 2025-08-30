package loss

// Forward computes mean squared error
func (mse) Forward(yPred, yTrue []float64) float64 {
	n := len(yPred)
	if n == 0 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		d := yPred[i] - yTrue[i]
		s += d * d
	}
	return s / float64(n)
}

// Backward computes gradient of MSE w.r.t. predictions
func (mse) Backward(yPred, yTrue []float64) []float64 {
	n := len(yPred)
	if n == 0 {
		return nil
	}
	grad := make([]float64, n)
	for i := 0; i < n; i++ {
		grad[i] = 2.0 * (yPred[i] - yTrue[i]) / float64(n)
	}
	return grad
}
