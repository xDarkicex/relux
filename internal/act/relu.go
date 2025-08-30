package act

type relu struct{}

func ReLU() Activation { return relu{} }

func (relu) Name() string { return "relu" }
func (relu) Apply(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (relu) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
