package layer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/relux/internal/compute"
)

type Layer interface {
	Forward(x []float64) []float64
	Backward(gradOut []float64, lr float64) []float64
	BackwardWithMomentum(gradOut []float64, lr, momentum, gradClip float64) []float64
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

	// Phase 5: Momentum velocities
	VW [][]float64 // Weight velocities
	VB []float64   // Bias velocities

	// Compute backend
	backend compute.ComputeBackend
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
		backend:   compute.NewComputeBackend(), // Auto-detect backend
	}
}

// Initialize momentum velocities
func (d *Dense) initMomentum() {
	if d.VW == nil {
		d.VW = make([][]float64, d.out)
		for j := 0; j < d.out; j++ {
			d.VW[j] = make([]float64, d.in)
		}
	}
	if d.VB == nil {
		d.VB = make([]float64, d.out)
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

	// Use compute backend for matrix multiplication
	// Convert weights to [][]float64 format
	weightsMatrix := make([][]float64, d.out)
	for j := 0; j < d.out; j++ {
		weightsMatrix[j] = make([]float64, d.in)
		copy(weightsMatrix[j], d.W[j])
	}

	// Input as 1x(in) matrix
	inputMatrix := [][]float64{x}

	// Accelerated matrix multiplication: [1×in] × [in×out] = [1×out]
	if d.backend != nil {
		if result, err := d.backend.MatMul(inputMatrix, weightsMatrix); err == nil && len(result) > 0 {
			// Extract result and add bias
			for j := 0; j < d.out; j++ {
				d.lastZ[j] = result[0][j] + d.B[j]
			}
		} else {
			// Fallback to original implementation
			d.computeForwardFallback(x)
		}
	} else {
		d.computeForwardFallback(x)
	}

	// Apply activation function
	if d.Act.Name() == "softmax" {
		return act.SoftmaxVec(d.lastZ)
	} else {
		// Try accelerated activation
		if d.backend != nil {
			if result, err := d.backend.ActivationFunc(d.Act.Name(), d.lastZ); err == nil {
				return result
			}
		}

		// Fallback to native activation
		out := make([]float64, d.out)
		for i, z := range d.lastZ {
			out[i] = d.Act.Apply(z)
		}
		return out
	}
}

func (d *Dense) ForwardBatch(inputs [][]float64) ([][]float64, error) {
	batchSize := len(inputs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	// Validate all inputs
	for i, x := range inputs {
		if len(x) != d.in {
			return nil, fmt.Errorf("input %d size mismatch: got %d, expected %d", i, len(x), d.in)
		}
	}

	// Use backend's enhanced ForwardBatch method (maintains all optimizations!)
	if d.backend != nil {
		return d.backend.ForwardBatch(inputs, d.W, d.B, d.Act.Name())
	}

	// Fallback to sequential processing if no backend available
	outputs := make([][]float64, batchSize)
	for i, input := range inputs {
		outputs[i] = d.Forward(input)
	}
	return outputs, nil
}

func (d *Dense) GetPerformanceInfo() map[string]interface{} {
	info := make(map[string]interface{})

	if d.backend != nil {
		info["backend"] = d.backend.Name()
		info["device_info"] = d.backend.DeviceInfo()
		info["available"] = d.backend.Available()

		// Use interface methods
		complexity := d.in * d.out
		info["layer_size"] = fmt.Sprintf("%d → %d", d.in, d.out)
		info["complexity"] = complexity
		info["will_use_gpu"] = d.backend.ShouldUseGPUForMatMul(1, d.in, d.out)
		info["activation_gpu"] = d.backend.ShouldUseGPUForActivation(d.out)
	}

	return info
}

func (d *Dense) computeForwardFallback(x []float64) {
	// Original implementation as fallback
	for j := 0; j < d.out; j++ {
		sum := d.B[j]
		wj := d.W[j]
		for i := 0; i < d.in; i++ {
			sum += wj[i] * x[i]
		}
		d.lastZ[j] = sum
	}
}

func (d *Dense) Backward(gradOut []float64, lr float64) []float64 {
	return d.BackwardWithMomentum(gradOut, lr, 0.0, 0.0)
}

func (d *Dense) BackwardWithMomentum(gradOut []float64, lr, momentum, gradClip float64) []float64 {
	// Initialize momentum if needed
	if momentum > 0 {
		d.initMomentum()
	}

	// Compute gradients w.r.t. pre-activation - handle softmax specially
	var gradZ []float64
	if d.Act.Name() == "softmax" {
		// For softmax, the gradient is already computed correctly in the loss function
		// when using categorical crossentropy, so we can use gradOut directly
		gradZ = make([]float64, len(gradOut))
		copy(gradZ, gradOut)
	} else {
		// For other activations, apply chain rule
		actDeriv := act.DerivativeVec(d.Act, d.lastZ)
		gradZ = make([]float64, d.out)
		for j := 0; j < d.out; j++ {
			gradZ[j] = gradOut[j] * actDeriv[j]
		}
	}

	// Gradient clipping
	if gradClip > 0 {
		gradNorm := 0.0
		for _, g := range gradZ {
			gradNorm += g * g
		}
		gradNorm = math.Sqrt(gradNorm)
		if gradNorm > gradClip {
			scale := gradClip / gradNorm
			for j := range gradZ {
				gradZ[j] *= scale
			}
		}
	}

	// Compute gradients w.r.t. input
	gradInput := make([]float64, d.in)
	for j := 0; j < d.out; j++ {
		for i := 0; i < d.in; i++ {
			gradInput[i] += d.W[j][i] * gradZ[j]
		}
	}

	// Update parameters with momentum
	for j := 0; j < d.out; j++ {
		// Update bias
		gradB := gradZ[j]
		if momentum > 0 {
			d.VB[j] = momentum*d.VB[j] - lr*gradB
			d.B[j] += d.VB[j]
		} else {
			d.B[j] -= lr * gradB
		}

		// Update weights
		for i := 0; i < d.in; i++ {
			gradW := gradZ[j] * d.lastInput[i]
			if momentum > 0 {
				d.VW[j][i] = momentum*d.VW[j][i] - lr*gradW
				d.W[j][i] += d.VW[j][i]
			} else {
				d.W[j][i] -= lr * gradW
			}
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

// Add cleanup method
func (d *Dense) Close() error {
	if d.backend != nil {
		return d.backend.Close()
	}
	return nil
}
