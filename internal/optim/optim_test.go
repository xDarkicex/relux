package optim_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/optim"
)

// bf16Of returns the bf16 bit pattern of x. Used in tests
// to build []uint16 weights from float64 literals.
func bf16Of(x float64) uint16 {
	// Match the optim package's float32ToBF16 implementation.
	u := math.Float32bits(float32(x))
	roundingBias := uint32(0x7FFF) + ((u >> 16) & 1)
	u += roundingBias
	return uint16(u >> 16)
}

func TestSGD_NoMomentum(t *testing.T) {
	opt := &optim.SGD{LR: 0.1}
	data := []uint16{bf16Of(1), bf16Of(2), bf16Of(3)}
	grad := []float32{0.1, 0.2, 0.3}
	if err := opt.Step([]optim.Param{{Name: "x", Data: data, Grad: grad}}); err != nil {
		t.Fatal(err)
	}
	// Each weight should decrease by lr*grad = 0.01, 0.02, 0.03.
	// Tolerance is bf16 precision (~0.78% relative).
	wantF := []float64{1 - 0.01, 2 - 0.02, 3 - 0.03}
	for i, v := range data {
		got := math.Float32frombits(uint32(v) << 16)
		if math.Abs(float64(got)-wantF[i]) > 1e-2 {
			t.Errorf("data[%d] = %v, want %v", i, got, wantF[i])
		}
	}
}

func TestSGD_Momentum(t *testing.T) {
	opt := &optim.SGD{LR: 0.1, Momentum: 0.9}
	data := []uint16{bf16Of(0), bf16Of(0)}
	grad := []float32{1, 1}
	// First step: vel = 0.9*0 + 1 = 1; data -= 0.1*1 = 0.1
	if err := opt.Step([]optim.Param{{Name: "x", Data: data, Grad: grad}}); err != nil {
		t.Fatal(err)
	}
	// Second step: vel = 0.9*1 + 1 = 1.9; data -= 0.1*1.9 = 0.19
	if err := opt.Step([]optim.Param{{Name: "x", Data: data, Grad: grad}}); err != nil {
		t.Fatal(err)
	}
	got := math.Float32frombits(uint32(data[0]) << 16)
	if math.Abs(float64(got)-(-0.29)) > 1e-2 {
		t.Errorf("after step 2: data[0] = %v, want -0.29", got)
	}
}

func TestSGD_Nesterov(t *testing.T) {
	opt := &optim.SGD{LR: 0.1, Momentum: 0.9, Nesterov: true}
	data := []uint16{bf16Of(0)}
	grad := []float32{1}
	// step 1: vel = 0.9*0 + 1 = 1; data -= 0.1*(0.9*1 + 1) = 0.1*(1.9) = 0.19
	if err := opt.Step([]optim.Param{{Name: "x", Data: data, Grad: grad}}); err != nil {
		t.Fatal(err)
	}
	got := math.Float32frombits(uint32(data[0]) << 16)
	if math.Abs(float64(got)-(-0.19)) > 1e-2 {
		t.Errorf("nesterov step 1: data = %v, want -0.19", got)
	}
}

func TestAdam_StepReducesLoss(t *testing.T) {
	// Minimize (x - 10)^2: gradient = 2(x-10)
	opt := &optim.Adam{LR: 0.5}
	data := []uint16{bf16Of(0)}
	for step := 0; step < 1000; step++ {
		got := math.Float32frombits(uint32(data[0]) << 16)
		g := 2 * (float64(got) - 10)
		if err := opt.Step([]optim.Param{{Name: "x", Data: data, Grad: []float32{float32(g)}}}); err != nil {
			t.Fatal(err)
		}
	}
	got := math.Float32frombits(uint32(data[0]) << 16)
	if math.Abs(float64(got)-10) > 0.5 {
		t.Errorf("adam did not converge: data[0] = %v, want ~10", got)
	}
}

func TestStateful_RoundTrip(t *testing.T) {
	sgd := &optim.SGD{LR: 0.1, Momentum: 0.9}
	if err := sgd.Step([]optim.Param{
		{Name: "w", Data: []uint16{bf16Of(0)}, Grad: []float32{1}},
		{Name: "b", Data: []uint16{bf16Of(0), bf16Of(0)}, Grad: []float32{0.5, 0.5}},
	}); err != nil {
		t.Fatal(err)
	}

	state := sgd.State()
	if state.Kind != "sgd" {
		t.Errorf("state.Kind = %q, want sgd", state.Kind)
	}
	if _, ok := state.Buffers["w"]; !ok {
		t.Error("state.Buffers missing \"w\"")
	}

	fresh := &optim.SGD{LR: 0.1, Momentum: 0.9}
	if err := fresh.LoadState(state); err != nil {
		t.Fatal(err)
	}
	if len(state.Buffers["w"]) != 1 {
		t.Errorf("vel w len = %d, want 1", len(state.Buffers["w"]))
	}
	if len(state.Buffers["b"]) != 2 {
		t.Errorf("vel b len = %d, want 2", len(state.Buffers["b"]))
	}
}

func TestAdam_StateRoundTrip(t *testing.T) {
	a := &optim.Adam{LR: 0.01}
	for i := 0; i < 5; i++ {
		if err := a.Step([]optim.Param{
			{Name: "w", Data: []uint16{bf16Of(0), bf16Of(0), bf16Of(0)}, Grad: []float32{0.1, 0.2, 0.3}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	state := a.State()
	if state.Kind != "adam" {
		t.Errorf("state.Kind = %q, want adam", state.Kind)
	}
	if state.Step != 5 {
		t.Errorf("state.Step = %d, want 5", state.Step)
	}
	for _, k := range []string{"m.w", "v.w"} {
		if _, ok := state.Buffers[k]; !ok {
			t.Errorf("state.Buffers missing %q", k)
		}
	}

	b := &optim.Adam{LR: 0.01}
	if err := b.LoadState(state); err != nil {
		t.Fatal(err)
	}
	state2 := b.State()
	if state2.Step != 5 {
		t.Errorf("reloaded state.Step = %d, want 5", state2.Step)
	}
	for k, buf := range state.Buffers {
		if len(buf) != len(state2.Buffers[k]) {
			t.Errorf("buffer %q length mismatch", k)
		}
		for i := range buf {
			if buf[i] != state2.Buffers[k][i] {
				t.Errorf("buffer %q[%d] = %v, want %v", k, i, buf[i], state2.Buffers[k][i])
			}
		}
	}
}

func TestClipGradNorm(t *testing.T) {
	params := []optim.Param{
		{Name: "a", Data: nil, Grad: []float32{3, 4}}, // norm 5
		{Name: "b", Data: nil, Grad: []float32{0, 0}}, // norm 0
	}
	got := optim.ClipGradNorm(params, 2.5)
	if math.Abs(float64(got)-5) > 1e-5 {
		t.Errorf("returned norm = %v, want 5", got)
	}
	// After clipping, every grad should be halved: norm becomes 2.5.
	got = optim.ClipGradNorm(params, 2.5)
	if math.Abs(float64(got)-2.5) > 1e-5 {
		t.Errorf("post-clip norm = %v, want 2.5", got)
	}

	// maxNorm larger than current norm: no rescaling, grad unchanged.
	params[0].Grad = []float32{3, 4}
	params[1].Grad = []float32{0, 0}
	optim.ClipGradNorm(params, 10)
	if params[0].Grad[0] != 3 || params[0].Grad[1] != 4 {
		t.Errorf("grad changed when norm < maxNorm: %v", params[0].Grad)
	}
}

func TestStep_LengthMismatch(t *testing.T) {
	opt := &optim.SGD{LR: 0.1}
	err := opt.Step([]optim.Param{
		{Name: "x", Data: []uint16{bf16Of(1), bf16Of(2), bf16Of(3)}, Grad: []float32{0.1}},
	})
	if err == nil {
		t.Fatal("expected length-mismatch error, got nil")
	}
}
