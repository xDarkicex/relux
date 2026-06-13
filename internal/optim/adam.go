// Package optim provides gradient-based parameter optimizers for training.
//
// The package decouples the update rule (SGD, Adam, ...) from the layer that
// owns the parameters. A layer exposes its trainable parameters as []Param,
// and an Optimizer applies accumulated gradients to those parameters in place.
//
// Usage:
//
//	// Build a network.
//	net, _ := relux.NewNetwork(...)
//
//	// Train with a custom optimizer.
//	net.Fit(X, Y,
//	    relux.Epochs(1000),
//	    relux.LearningRate(0.001),
//	    relux.Optimizer(&optim.Adam{LR: 0.001}),
//	)
//
// Or stand-alone:
//
//	opt := &optim.SGD{LR: 0.01, Momentum: 0.9}
//	params := layer.Params()           // []optim.Param
//	opt.Step(params)
//
// The Stateful interface lets optimizers with internal state (momentum
// buffers, Adam moment estimates) be serialized alongside the model.
package optim

import (
	"fmt"
	"math"
)

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
//
// Adam maintains exponential moving averages of the gradient (first moment)
// and of the squared gradient (second moment), with bias correction:
//
//	m_t     = beta1 * m_{t-1} + (1 - beta1) * grad
//	v_t     = beta2 * v_{t-1} + (1 - beta2) * grad²
//	m_hat   = m_t / (1 - beta1^t)
//	v_hat   = v_t / (1 - beta2^t)
//	param  -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Defaults: LR=0.001, Beta1=0.9, Beta2=0.999, Eps=1e-8. Adam is generally
// robust to learning rate and works well on noisy or sparse gradients.
//
// State precision: Adam's m and v are stored in float32. This is critical
// — bf16 truncation of these running averages (which are exponentially
// decaying sums of tiny gradients with beta2=0.999) destroys the
// precision needed for the variance estimate. Storing in float64 wastes
// memory and the math is well-conditioned for float32.
type Adam struct {
	LR    float32
	Beta1 float32
	Beta2 float32
	Eps   float32

	// t is the timestep, incremented on each Step call.
	t int

	// m and v are the first and second moment estimates, keyed by Param.Name.
	m, v map[string][]float32
}

func (a *Adam) init() {
	if a.LR == 0 {
		a.LR = 0.001
	}
	if a.Beta1 == 0 {
		a.Beta1 = 0.9
	}
	if a.Beta2 == 0 {
		a.Beta2 = 0.999
	}
	if a.Eps == 0 {
		a.Eps = 1e-8
	}
	if a.m == nil {
		a.m = make(map[string][]float32)
	}
	if a.v == nil {
		a.v = make(map[string][]float32)
	}
}

func (a *Adam) Step(params []Param) error {
	a.init()
	a.t++

	bc1 := 1.0 - float32(math.Pow(float64(a.Beta1), float64(a.t)))
	bc2 := 1.0 - float32(math.Pow(float64(a.Beta2), float64(a.t)))

	for _, p := range params {
		if len(p.Data) != len(p.Grad) {
			return fmt.Errorf("adam: param %q: data/grad length mismatch (%d vs %d)",
				p.Name, len(p.Data), len(p.Grad))
		}

		mBuf, exists := a.m[p.Name]
		if !exists || len(mBuf) != len(p.Data) {
			mBuf = make([]float32, len(p.Data))
			a.m[p.Name] = mBuf
		}
		vBuf, exists := a.v[p.Name]
		if !exists || len(vBuf) != len(p.Data) {
			vBuf = make([]float32, len(p.Data))
			a.v[p.Name] = vBuf
		}

		lr := a.LR
		b1 := a.Beta1
		b2 := a.Beta2
		eps := a.Eps
		oneMinusB1 := 1.0 - b1
		oneMinusB2 := 1.0 - b2
		for i := range p.Data {
			g := p.Grad[i]
			mBuf[i] = b1*mBuf[i] + oneMinusB1*g
			vBuf[i] = b2*vBuf[i] + oneMinusB2*g*g
			mHat := mBuf[i] / bc1
			vHat := vBuf[i] / bc2
			// Update is computed in float32 (so it accumulates
			// at the same precision as the gradient) and
			// stored as bf16. The bf16 downcast truncates the
			// mantissa but preserves the exponent — the
			// truncation is well within the gradient noise
			// floor for typical transformer params.
			step := lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
			p.Data[i] = bf16AddFloat32(bf16ToFloat32(p.Data[i]), -step)
		}
	}
	return nil
}

func (a *Adam) State() State {
	buffers := make(map[string][]float32, 2*len(a.m))
	for k, v := range a.m {
		buf := make([]float32, len(v))
		copy(buf, v)
		buffers["m."+k] = buf
	}
	for k, v := range a.v {
		buf := make([]float32, len(v))
		copy(buf, v)
		buffers["v."+k] = buf
	}
	return State{Kind: "adam", Buffers: buffers, Step: a.t}
}

func (a *Adam) LoadState(state State) error {
	if state.Kind != "" && state.Kind != "adam" {
		return fmt.Errorf("adam: cannot load state of kind %q", state.Kind)
	}
	a.t = state.Step
	a.m = make(map[string][]float32)
	a.v = make(map[string][]float32)
	for k, buf := range state.Buffers {
		switch {
		case len(k) > 2 && k[:2] == "m.":
			cp := make([]float32, len(buf))
			copy(cp, buf)
			a.m[k[2:]] = cp
		case len(k) > 2 && k[:2] == "v.":
			cp := make([]float32, len(buf))
			copy(cp, buf)
			a.v[k[2:]] = cp
		}
	}
	return nil
}
