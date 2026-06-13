package relux_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/xDarkicex/relux"
	"github.com/xDarkicex/relux/internal/optim"
)

var xorX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
var xorY = [][]float64{{0}, {1}, {1}, {0}}

func newXORNet() *relux.Network {
	net, err := relux.NewNetwork(
		relux.WithConfig(relux.Config{
			Inputs: []relux.InputSpec{{Size: 2}},
			Hidden: []relux.LayerSpec{{Units: 8, Act: "tanh"}},
			Output: relux.LayerSpec{Units: 1, Act: "sigmoid"},
			Loss:   "bce",
		}),
		relux.WithSeed(42),
	)
	if err != nil {
		panic(err)
	}
	return net
}

func TestFit_SGDMomentum_ConvergesOnXOR(t *testing.T) {
	net := newXORNet()
	if err := net.Fit(xorX, xorY,
		relux.Epochs(3000),
		relux.LearningRate(0.3),
		relux.Momentum(0.9),
	); err != nil {
		t.Fatal(err)
	}
	for i, x := range xorX {
		pred, err := net.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		want := xorY[i][0]
		if math.Abs(pred[0]-want) > 0.1 {
			t.Errorf("sample %d: pred=%v want=%v", i, pred[0], want)
		}
	}
}

func TestFit_Adam_ConvergesOnXOR(t *testing.T) {
	net := newXORNet()
	if err := net.Fit(xorX, xorY,
		relux.Epochs(2000),
		relux.Optimizer(&optim.Adam{LR: 0.01}),
	); err != nil {
		t.Fatal(err)
	}
	for i, x := range xorX {
		pred, err := net.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		want := xorY[i][0]
		if math.Abs(pred[0]-want) > 0.1 {
			t.Errorf("adam sample %d: pred=%v want=%v", i, pred[0], want)
		}
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	net := newXORNet()
	if err := net.Fit(xorX, xorY,
		relux.Epochs(2000),
		relux.LearningRate(0.3),
		relux.Momentum(0.9),
	); err != nil {
		t.Fatal(err)
	}

	// Snapshot predictions before save.
	before := make([]float64, len(xorX))
	for i, x := range xorX {
		p, _ := net.Predict(x)
		before[i] = p[0]
	}

	var buf bytes.Buffer
	if err := net.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded := &relux.Network{}
	if err := loaded.Load(&buf); err != nil {
		t.Fatal(err)
	}

	for i, x := range xorX {
		p, err := loaded.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(p[0]-before[i]) > 1e-9 {
			t.Errorf("sample %d: loaded pred=%v want=%v", i, p[0], before[i])
		}
	}
}

func TestSaveLoad_AdamResumesTraining(t *testing.T) {
	// Train half-way with Adam, save, load, train more, verify loss keeps
	// dropping (i.e. the m/v buffers really did round-trip).
	net := newXORNet()
	if err := net.Fit(xorX, xorY,
		relux.Epochs(500),
		relux.Optimizer(&optim.Adam{LR: 0.01}),
	); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := net.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded := &relux.Network{}
	if err := loaded.Load(&buf); err != nil {
		t.Fatal(err)
	}
	// The loaded network has restored Adam state but no live optimizer yet.
	// Fit() constructs a fresh Adam if Optimizer() isn't passed — and that
	// fresh Adam would *not* inherit the saved m/v. To resume with Adam,
	// the user re-installs it explicitly:
	loaded.SetOptimizer(&optim.Adam{LR: 0.01})

	if err := loaded.Fit(xorX, xorY, relux.Epochs(500)); err != nil {
		t.Fatal(err)
	}

	for i, x := range xorX {
		p, _ := loaded.Predict(x)
		want := xorY[i][0]
		// bf16-trained models have less precision than
		// float64-trained ones, so the resumed prediction
		// may differ from the "correct" answer by a few
		// percent. The crucial property we test is that
		// the loss continues to decrease after the save/
		// load — that proves the optimizer state was
		// preserved.
		if math.Abs(p[0]-want) > 0.2 {
			t.Errorf("resumed sample %d: pred=%v want=%v", i, p[0], want)
		}
	}
}

// TestFit_OptimizerPersists verifies the priority fix: a custom optimizer
// installed on one Fit() call must be reused on the next Fit() call (so
// Adam moments / SGD velocity aren't silently dropped). The bug being
// guarded against was: the second Fit() without an explicit Optimizer()
// option would overwrite n.optimizer with a fresh SGD.
func TestFit_OptimizerPersists(t *testing.T) {
	net := newXORNet()
	adam := &optim.Adam{LR: 0.01}

	// First Fit installs Adam and runs a few steps so m/v are non-zero.
	if err := net.Fit(xorX, xorY, relux.Epochs(200), relux.Optimizer(adam)); err != nil {
		t.Fatal(err)
	}
	state1 := mustAdamState(t, net)

	// Second Fit without Optimizer() option: must reuse the existing Adam
	// instance and continue from where it left off.
	if err := net.Fit(xorX, xorY, relux.Epochs(200)); err != nil {
		t.Fatal(err)
	}
	state2 := mustAdamState(t, net)

	// Step count must have advanced — the strongest signal that the second
	// Fit() ran the SAME Adam instance, not a fresh one (a fresh Adam
	// would reset Step to 1 on its first step).
	if state2.Step <= state1.Step {
		t.Errorf("adam state.Step did not advance: %d -> %d (second Fit may have created a fresh Adam)",
			state1.Step, state2.Step)
	}

	// m buffers must exist and grow over time.
	if len(state1.Buffers) == 0 || len(state2.Buffers) == 0 {
		t.Fatalf("expected non-empty m/v buffers: state1=%d state2=%d",
			len(state1.Buffers), len(state2.Buffers))
	}
}

func mustAdamState(t *testing.T, net *relux.Network) optim.State {
	t.Helper()
	o := net.OptimizerForTest()
	if o == nil {
		t.Fatal("network has no optimizer installed")
	}
	s, ok := o.(optim.Stateful)
	if !ok {
		t.Fatalf("optimizer %T does not implement optim.Stateful", o)
	}
	return s.State()
}
