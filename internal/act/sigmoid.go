package act

import "math"

type sigmoid struct{}

func Sigmoid() Activation { return sigmoid{} }

func (sigmoid) Name() string { return "sigmoid" }
func (sigmoid) Apply(x float64) float64 {
	if x > 709 { // Prevent overflow
		return 1.0
	}
	if x < -709 { // Prevent underflow
		return 0.0
	}
	return 1 / (1 + math.Exp(-x))
}

func (sigmoid) Derivative(x float64) float64 {
	s := sigmoid{}.Apply(x) // Use stable version
	return s * (1 - s)
}
