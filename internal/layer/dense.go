package layer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/xDarkicex/relux/internal/act"
	"github.com/xDarkicex/relux/internal/alloc"
	"github.com/xDarkicex/relux/internal/compute"
	"github.com/xDarkicex/relux/internal/optim"
)

// Layer is a trainable stage of a network. Implementations are expected to be
// stateful in the sense that Backward populates per-parameter gradients
// (exposed via Params) and Forward caches the inputs needed to compute them.
//
// The data interface is float64 — the activations and data pipeline use
// float64. The weights inside each layer are stored as bfloat16 (per the
// optim.Param contract) and the gradients as float32. The math
// (matmul) happens in float32: weights are widened from bf16 to f32
// on the fly, and outputs are float32. This gives 16-bit weight
// storage and bf16 matmul capability on accelerators, while
// preserving float64 user-facing data APIs.
type Layer interface {
	Forward(x []float64) []float64
	Backward(gradOut []float64) []float64
	Params() []optim.Param
}

// Dense is a fully connected (a.k.a. linear) layer:
//
//	y = activation(x @ Wᵀ + b)
//
// W is stored flat in row-major order as bfloat16: W[j*in + i] is
// the weight from input unit i to output unit j. The flat layout is
// friendlier to in-place updates by the optimizer and to the
// matmul backend's cache access pattern.
//
// bfloat16 storage gives 2x memory and bandwidth vs float32. The
// matmul happens in float32: each bf16 weight is widened to f32 on
// the fly, multiplied with the float32-cast input, and the result
// accumulated in f32. The output is widened to float64 to match
// the user-facing data API.
type Dense struct {
	W   []uint16 // bfloat16
	B   []uint16 // bfloat16
	Act act.Activation
	in  int
	out int

	id int

	lastInput []float32 // float32 working copy of input X for the matmul
	lastZ     []float32 // float32 pre-activation for the activation derivative

	gradW []float32
	gradB []float32

	backend compute.ComputeBackend
}

// NewDense creates a Dense layer with He (relu) or Xavier (other) init.
func NewDense(in, out int, a act.Activation, rnd *rand.Rand) *Dense {
	if rnd == nil {
		rnd = rand.New(rand.NewSource(1))
	}
	if a == nil {
		a = act.Identity()
	}

	var limit float64
	switch a.Name() {
	case "relu":
		limit = math.Sqrt(2.0 / float64(in))
	default:
		limit = math.Sqrt(6.0 / float64(in+out))
	}

	wF32 := alloc.Float32(out * in)
	for j := 0; j < out; j++ {
		for i := 0; i < in; i++ {
			if a.Name() == "relu" {
				wF32[j*in+i] = float32(rnd.NormFloat64() * limit)
			} else {
				wF32[j*in+i] = float32(rnd.Float64()*2*limit - limit)
			}
		}
	}
	w := alloc.Uint16(out * in)
	for i, x := range wF32 {
		w[i] = float32ToBF16(x)
	}
	bF32 := alloc.Float32(out)
	b := alloc.Uint16(out)
	for i, x := range bF32 {
		_ = x
		b[i] = float32ToBF16(0)
	}

	return &Dense{
		W:         w,
		B:         b,
		Act:       a,
		in:        in,
		out:       out,
		lastInput: alloc.Float32(in),
		lastZ:     alloc.Float32(out),
		gradW:     alloc.Float32(out * in),
		gradB:     alloc.Float32(out),
		backend:   compute.NewComputeBackend(),
	}
}

// OutputSize returns the output dimension.
func (d *Dense) OutputSize() int { return d.out }

// In returns the input dimension.
func (d *Dense) In() int { return d.in }

// Out returns the output dimension.
func (d *Dense) Out() int { return d.out }

// Params returns the layer's trainable parameters together with their
// current gradients. The returned slice is freshly allocated on every call
// and is safe to pass to an optimizer. Optimizers must not cache the slice
// across Step calls (the underlying arrays are owned by the layer).
func (d *Dense) Params() []optim.Param {
	return []optim.Param{
		{Name: fmt.Sprintf("L%d.W", d.id), Data: d.W, Grad: d.gradW},
		{Name: fmt.Sprintf("L%d.B", d.id), Data: d.B, Grad: d.gradB},
	}
}

// SetLayerID assigns a stable index to the layer for parameter naming.
// Network.buildFrom calls this for every layer it appends so that
// optimizer state keyed by param name does not collide between layers.
func (d *Dense) SetLayerID(id int) { d.id = id }

// Forward computes the layer's output and caches the input and pre-activation
// values for the subsequent Backward call.
func (d *Dense) Forward(x []float64) []float64 {
	if len(x) != d.in {
		panic(fmt.Sprintf("input size mismatch: got %d, expected %d", len(x), d.in))
	}
	// Cast input to float32 for the matmul.
	for i, v := range x {
		d.lastInput[i] = float32(v)
	}

	// Compute pre-activation z = x @ W^T + b in float32.
	// Widen bf16 weights on the fly.
	for j := 0; j < d.out; j++ {
		sum := bf16ToFloat32(d.B[j])
		rowStart := j * d.in
		for i := 0; i < d.in; i++ {
			sum += bf16ToFloat32(d.W[rowStart+i]) * d.lastInput[i]
		}
		d.lastZ[j] = sum
	}

	// Apply activation in float32, then widen to float64 for the API.
	zF64 := make([]float64, d.out)
	for i, v := range d.lastZ {
		zF64[i] = float64(v)
	}
	if d.Act.Name() == "softmax" {
		return act.SoftmaxVec(zF64)
	}
	out := make([]float64, d.out)
	for i, z := range zF64 {
		out[i] = d.Act.Apply(z)
	}
	return out
}

// ForwardBatch runs the layer over a batch.
func (d *Dense) ForwardBatch(inputs [][]float64) ([][]float64, error) {
	batchSize := len(inputs)
	if batchSize == 0 {
		return nil, fmt.Errorf("empty batch")
	}
	for i, x := range inputs {
		if len(x) != d.in {
			return nil, fmt.Errorf("input %d size mismatch: got %d, expected %d", i, len(x), d.in)
		}
	}
	if d.backend != nil {
		// Backend path: build bf16 weights and float32 input matrix.
		// For simplicity, fall back to per-sample Forward for now.
		// (The backend's float64 matmul is replaced by our float32
		// path which is already efficient on small layers.)
	}
	outputs := make([][]float64, batchSize)
	for i, input := range inputs {
		outputs[i] = d.Forward(input)
	}
	return outputs, nil
}

// Backward computes the gradient of the loss w.r.t. the layer's input,
// populates the per-parameter gradient buffers (exposed via Params), and
// does not touch the parameters themselves — that is the optimizer's job.
//
// For softmax activations, the upstream loss is expected to have already
// applied the softmax + cross-entropy fused gradient (see internal/loss).
// In that case gradOut is taken as grad w.r.t. the pre-softmax logits and the
// activation derivative step is skipped.
//
// All math here is float32: gradW and gradB are stored as float32 per
// the optim.Param.Grad contract. gradInput is widened to float64 for
// the user-facing API.
func (d *Dense) Backward(gradOut []float64) []float64 {
	if len(gradOut) != d.out {
		panic(fmt.Sprintf("grad size mismatch: got %d, expected %d", len(gradOut), d.out))
	}

	// Cast gradOut to float32.
	gOut := make([]float32, d.out)
	for i, v := range gradOut {
		gOut[i] = float32(v)
	}
	gradZ := make([]float32, d.out)
	if d.Act.Name() == "softmax" {
		copy(gradZ, gOut)
	} else {
		for j := 0; j < d.out; j++ {
			gradZ[j] = gOut[j] * float32(d.Act.Derivative(float64(d.lastZ[j])))
		}
	}

	// gradW[j, i] = gradZ[j] * lastInput[i]
	// gradB[j]   = gradZ[j]
	for j := 0; j < d.out; j++ {
		d.gradB[j] = gradZ[j]
		rowStart := j * d.in
		for i := 0; i < d.in; i++ {
			d.gradW[rowStart+i] = gradZ[j] * d.lastInput[i]
		}
	}

	// gradInput[i] = sum_j W[j, i] * gradZ[j]
	gradInput := make([]float64, d.in)
	for j := 0; j < d.out; j++ {
		g := gradZ[j]
		rowStart := j * d.in
		for i := 0; i < d.in; i++ {
			gradInput[i] += float64(bf16ToFloat32(d.W[rowStart+i]) * g)
		}
	}
	return gradInput
}

// GetPerformanceInfo returns diagnostic info about the layer's compute path.
func (d *Dense) GetPerformanceInfo() map[string]interface{} {
	info := make(map[string]interface{})
	if d.backend != nil {
		info["backend"] = d.backend.Name()
		info["device_info"] = d.backend.DeviceInfo()
		info["available"] = d.backend.Available()
		info["layer_size"] = fmt.Sprintf("%d → %d", d.in, d.out)
		info["complexity"] = d.in * d.out
	}
	return info
}

// InitializeCache resizes the per-call caches and gradient buffers. Used by
// the serializer when restoring a layer from a snapshot.
func (d *Dense) InitializeCache(inputSize, outputSize int) {
	d.in = inputSize
	d.out = outputSize
	d.lastInput = alloc.Float32(inputSize)
	d.lastZ = alloc.Float32(outputSize)
	d.gradW = alloc.Float32(outputSize * inputSize)
	d.gradB = alloc.Float32(outputSize)
}

// Close releases the layer's compute backend.
func (d *Dense) Close() error {
	if d.backend != nil {
		return d.backend.Close()
	}
	return nil
}
