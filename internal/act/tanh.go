package act

import "math"

type tanh struct{}

func Tanh() Activation { return tanh{} }

func (tanh) Name() string { return "tanh" }
func (tanh) Apply(x float64) float64 {
	return math.Tanh(x)
}

func (tanh) Derivative(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}
