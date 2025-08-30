package layer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/xDarkicex/relux/internal/act"
)

type Layer interface {
	Forward(x []float64) []float64
	Backward(gradOut []float64, lr float64) []float64
}

type Dense struct {
	W   [][]float64
	B   []float64
	Act act.Activation
	in  int
	out int

	// Cache for backprop
	lastInput []float64
	lastZ     []float64 // pre-activation values
}

func NewDense(in, out int, a act.Activation, rnd *rand.Rand) *Dense {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(1))
	}
	if a == nil {
		a = act.Identity()
	}
	W := make([][]float64, out)
	B := make([]float64, out)

	// Enhanced weight initialization
	var limit float64
	switch a.Name() {
	case "relu":
		// He initialization for ReLU
		limit = math.Sqrt(2.0 / float64(in))
	case "tanh":
		// Xavier for tanh
		limit = math.Sqrt(6.0 / float64(in+out))
	default:
		// Xavier for sigmoid and others
		limit = math.Sqrt(6.0 / float64(in+out))
	}

	for j := 0; j < out; j++ {
		W[j] = make([]float64, in)
		for i := 0; i < in; i++ {
			if a.Name() == "relu" {
				// He: normal distribution * sqrt(2/fan_in)
				W[j][i] = rnd.NormFloat64() * limit
			} else {
				// Xavier: uniform in [-limit, limit]
				W[j][i] = (rnd.Float64()*2*limit - limit)
			}
		}
		B[j] = 0
	}

	return &Dense{
		W: W, B: B, Act: a, in: in, out: out,
		lastInput: make([]float64, in),
		lastZ:     make([]float64, out),
	}
}

// OutputSize returns the output dimension of this layer
func (d *Dense) OutputSize() int {
	return d.out
}

func (d *Dense) Forward(x []float64) []float64 {
	if len(x) != d.in {
		panic(fmt.Sprintf("input size mismatch: got %d, expected %d", len(x), d.in))
	}

	// Cache input for backprop
	copy(d.lastInput, x)

	// Reuse lastZ slice for pre-activation
	for j := 0; j < d.out; j++ {
		sum := d.B[j]
		wj := d.W[j]
		for i := 0; i < d.in; i++ {
			sum += wj[i] * x[i]
		}
		d.lastZ[j] = sum
	}

	// Apply activation in-place to a new slice (don't modify lastZ)
	out := make([]float64, d.out)
	for i, z := range d.lastZ {
		out[i] = d.Act.Apply(z)
	}
	return out
}

func (d *Dense) Backward(gradOut []float64, lr float64) []float64 {
	// Compute gradients w.r.t. pre-activation (z)
	actDeriv := act.DerivativeVec(d.Act, d.lastZ)
	gradZ := make([]float64, d.out)
	for j := 0; j < d.out; j++ {
		gradZ[j] = gradOut[j] * actDeriv[j]
	}

	// Gradient clipping for stability
	const maxGradNorm = 5.0
	gradNorm := 0.0
	for _, g := range gradZ {
		gradNorm += g * g
	}
	gradNorm = math.Sqrt(gradNorm)

	if gradNorm > maxGradNorm {
		scale := maxGradNorm / gradNorm
		for j := range gradZ {
			gradZ[j] *= scale
		}
	}

	// Compute gradients w.r.t. input (for previous layer)
	gradInput := make([]float64, d.in)
	for j := 0; j < d.out; j++ {
		for i := 0; i < d.in; i++ {
			gradInput[i] += d.W[j][i] * gradZ[j]
		}
	}

	// Update parameters (SGD)
	for j := 0; j < d.out; j++ {
		// Update bias
		d.B[j] -= lr * gradZ[j]

		// Update weights
		for i := 0; i < d.in; i++ {
			gradW := gradZ[j] * d.lastInput[i]
			d.W[j][i] -= lr * gradW
		}
	}

	return gradInput
}

// InitializeCache initializes the cache slices for backpropagation.
func (d *Dense) InitializeCache(inputSize, outputSize int) {
	d.in = inputSize
	d.out = outputSize
	d.lastInput = make([]float64, inputSize)
	d.lastZ = make([]float64, outputSize)
}
