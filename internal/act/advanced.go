package act

import "math"

// Swish activation (x * sigmoid(x)) - popular in modern architectures
type swish struct{}

func Swish() Activation { return swish{} }

func (swish) Name() string { return "swish" }

func (swish) Apply(x float64) float64 {
	sigmoid := 1.0 / (1.0 + math.Exp(-x))
	return x * sigmoid
}

func (swish) Derivative(x float64) float64 {
	sigmoid := 1.0 / (1.0 + math.Exp(-x))
	return sigmoid + x*sigmoid*(1-sigmoid)
}

// GELU activation - used in transformers
type gelu struct{}

func GELU() Activation { return gelu{} }

func (gelu) Name() string { return "gelu" }

func (gelu) Apply(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

func (gelu) Derivative(x float64) float64 {
	// Approximate derivative
	cdf := 0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
	pdf := math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
	return cdf + x*pdf
}

// LeakyReLU activation
type leakyReLU struct {
	alpha float64
}

func LeakyReLU(alpha float64) Activation {
	if alpha <= 0 {
		alpha = 0.01 // Default leaky coefficient
	}
	return leakyReLU{alpha: alpha}
}

func (l leakyReLU) Name() string { return "leaky_relu" }

func (l leakyReLU) Apply(x float64) float64 {
	if x > 0 {
		return x
	}
	return l.alpha * x
}

func (l leakyReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return l.alpha
}
