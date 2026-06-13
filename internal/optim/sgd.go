package optim

import "fmt"

// SGD implements stochastic gradient descent with optional momentum.
//
// Update rule without momentum:
//
//	param -= lr * grad
//
// Update rule with momentum (Polyak / "heavy ball"):
//
//	velocity = momentum * velocity + grad
//	param -= lr * velocity
//
// Momentum in [0, 1). Set to 0 for plain SGD. Values around 0.9 are common
// for MLPs.
//
// When Nesterov is true, the gradient is applied to the lookahead position:
//
//	velocity = momentum * velocity + grad
//	param -= lr * (momentum * velocity + grad)
//
// The math operates in float32. Weights (Data) are bf16; the
// update is computed at float32 precision and downcast on
// write. Velocities are kept in float32 — the running average
// is sensitive to the same precision issues as Adam's m.
type SGD struct {
	LR       float32
	Momentum float32
	Nesterov bool

	// velocities stores the per-parameter velocity buffers,
	// keyed by Param.Name. Lazily allocated on first use.
	velocities map[string][]float32
}

func (s *SGD) Step(params []Param) error {
	if s.velocities == nil {
		s.velocities = make(map[string][]float32)
	}

	for _, p := range params {
		if len(p.Data) != len(p.Grad) {
			return fmt.Errorf("sgd: param %q: data/grad length mismatch (%d vs %d)",
				p.Name, len(p.Data), len(p.Grad))
		}

		vel, exists := s.velocities[p.Name]
		if !exists || len(vel) != len(p.Data) {
			vel = make([]float32, len(p.Data))
			s.velocities[p.Name] = vel
		}

		switch {
		case s.Momentum == 0:
			for i := range p.Data {
				p.Data[i] = bf16AddFloat32(bf16ToFloat32(p.Data[i]), -s.LR*p.Grad[i])
			}
		case s.Nesterov:
			mom := s.Momentum
			lr := s.LR
			for i := range p.Data {
				vel[i] = mom*vel[i] + p.Grad[i]
				p.Data[i] = bf16AddFloat32(bf16ToFloat32(p.Data[i]), -lr*(mom*vel[i]+p.Grad[i]))
			}
		default:
			mom := s.Momentum
			lr := s.LR
			for i := range p.Data {
				vel[i] = mom*vel[i] + p.Grad[i]
				p.Data[i] = bf16AddFloat32(bf16ToFloat32(p.Data[i]), -lr*vel[i])
			}
		}
	}
	return nil
}

func (s *SGD) State() State {
	buffers := make(map[string][]float32, len(s.velocities))
	for k, v := range s.velocities {
		buf := make([]float32, len(v))
		copy(buf, v)
		buffers[k] = buf
	}
	return State{Kind: "sgd", Buffers: buffers, Step: 0}
}

func (s *SGD) LoadState(state State) error {
	if state.Kind != "" && state.Kind != "sgd" {
		return fmt.Errorf("sgd: cannot load state of kind %q", state.Kind)
	}
	if len(state.Buffers) == 0 {
		s.velocities = nil
		return nil
	}
	s.velocities = make(map[string][]float32, len(state.Buffers))
	for k, v := range state.Buffers {
		buf := make([]float32, len(v))
		copy(buf, v)
		s.velocities[k] = buf
	}
	return nil
}
