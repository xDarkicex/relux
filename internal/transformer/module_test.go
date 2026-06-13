package transformer_test

import (
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

// fakeModule is the minimal Module impl for testing the
// Mode / SetMode contract via BaseModule.
type fakeModule struct {
	transformer.BaseModule
	forwardCalls int
}

func (f *fakeModule) Forward(x *transformer.Tensor) *transformer.Tensor {
	f.forwardCalls++
	return x.Clone()
}
func (f *fakeModule) Backward(g *transformer.Tensor) *transformer.Tensor { return g.Clone() }
func (f *fakeModule) Params() []struct{}                                 { return nil }

func TestMode_String(t *testing.T) {
	if transformer.Train.String() != "train" {
		t.Errorf("Train.String() = %q, want train", transformer.Train.String())
	}
	if transformer.Inference.String() != "inference" {
		t.Errorf("Inference.String() = %q, want inference", transformer.Inference.String())
	}
}

func TestBaseModule_DefaultIsTrain(t *testing.T) {
	f := &fakeModule{}
	if f.Mode() != transformer.Train {
		t.Errorf("default mode = %s, want train", f.Mode())
	}
}

func TestBaseModule_SetMode(t *testing.T) {
	f := &fakeModule{}
	f.SetMode(transformer.Inference)
	if f.Mode() != transformer.Inference {
		t.Errorf("after SetMode(Inference), mode = %s, want inference", f.Mode())
	}
	f.SetMode(transformer.Train)
	if f.Mode() != transformer.Train {
		t.Errorf("after SetMode(Train), mode = %s, want train", f.Mode())
	}
}
