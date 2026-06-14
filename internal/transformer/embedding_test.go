package transformer_test

import (
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestEmbedding_Gather(t *testing.T) {
	e := transformer.NewEmbedding(4, 3)
	// Override the weight to known values. Data is bf16; we
	// cast from int for exact representation.
	for i := range e.Params()[0].Data {
		e.Params()[0].Data[i] = transformer.BF16FromF32(float32(i + 1))
	}
	// ids = [1, 3] -> output rows = weight[1, :] = [4,5,6]
	// and weight[3, :] = [10,11,12]
	out := e.Forward([]int{1, 3}, 1)
	got := out.DataF32()
	want := []float32{4, 5, 6, 10, 11, 12}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestEmbedding_BackwardScatter(t *testing.T) {
	e := transformer.NewEmbedding(4, 3)
	// Forward with ids [0, 2, 0] — token 0 appears twice.
	ids := []int{0, 2, 0}
	_ = e.Forward(ids, 1)

	// gradOut: [3, 3] with rows = [1,2,3, 4,5,6, 7,8,9]
	gradOut := transformer.NewTensor(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
		1, 3, 3,
	)
	e.Backward(gradOut)

	// After scatter: weight.Grad[0, :] += [1,2,3] + [7,8,9] = [8,10,12]
	// weight.Grad[1, :] unchanged (not used)
	// weight.Grad[2, :] += [4,5,6] = [4,5,6]
	// weight.Grad[3, :] unchanged (not used)
	want := []float32{8, 10, 12, 0, 0, 0, 4, 5, 6, 0, 0, 0}
	for i, v := range e.Params()[0].Grad {
		if v != want[i] {
			t.Errorf("grad[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestEmbedding_OutOfRangePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on out-of-range token id")
		}
	}()
	e := transformer.NewEmbedding(4, 3)
	e.Forward([]int{0, 5}, 1) // 5 is out of range
}

func TestEmbedding_NoParamsAreInputGrad(t *testing.T) {
	e := transformer.NewEmbedding(4, 3)
	_ = e.Forward([]int{0}, 1)
	gradOut := transformer.NewTensor([]float32{1, 2, 3}, 1, 1, 3)
	if grad := e.Backward(gradOut); grad != nil {
		t.Errorf("Embedding.Backward should return nil (no input grad), got %v", grad)
	}
}
