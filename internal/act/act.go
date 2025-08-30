package act

type Activation interface {
	Name() string
	Apply(x float64) float64
	Derivative(x float64) float64
}

// Identity activation for linear (no activation)
type identity struct{}

func Identity() Activation {
	return identity{}
}

func (identity) Name() string                 { return "identity" }
func (identity) Apply(x float64) float64      { return x }
func (identity) Derivative(x float64) float64 { return 1.0 }

// ApplyVec applies activation function to a vector of values
func ApplyVec(a Activation, z []float64) []float64 {
	out := make([]float64, len(z))
	for i, v := range z {
		out[i] = a.Apply(v)
	}
	return out
}

// DerivativeVec applies derivative function to a vector of values
func DerivativeVec(a Activation, z []float64) []float64 {
	out := make([]float64, len(z))
	for i, v := range z {
		out[i] = a.Derivative(v)
	}
	return out
}
