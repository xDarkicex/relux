package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestTensor_NewAndAccessors(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	t1 := transformer.NewTensor(data, 2, 3)
	if t1.Rank() != 2 {
		t.Errorf("Rank = %d, want 2", t1.Rank())
	}
	if t1.Size() != 6 {
		t.Errorf("Size = %d, want 6", t1.Size())
	}
	if t1.DType() != transformer.Float32 {
		t.Errorf("DType = %s, want f32", t1.DType())
	}
	got := t1.DataF32()
	if len(got) != 6 {
		t.Errorf("DataF32 len = %d, want 6", len(got))
	}
	for i, v := range got {
		if v != float32(i+1) {
			t.Errorf("DataF32[%d] = %v, want %v", i, v, i+1)
		}
	}
	shape := t1.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("Shape = %v, want [2 3]", shape)
	}
}

func TestTensor_NewPanicsOnSizeMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on size mismatch")
		}
	}()
	transformer.NewTensor([]float32{1, 2, 3}, 2, 3) // 3 elements, shape says 6
}

func TestTensor_ZerosAndFromBF16(t *testing.T) {
	z := transformer.ZerosF32(2, 4)
	if z.DType() != transformer.Float32 {
		t.Errorf("ZerosF32 dtype = %s, want f32", z.DType())
	}
	for i, v := range z.DataF32() {
		if v != 0 {
			t.Errorf("ZerosF32[%d] = %v, want 0", i, v)
		}
	}

	zb := transformer.ZerosBF16(3, 5)
	if zb.DType() != transformer.BFloat16 {
		t.Errorf("ZerosBF16 dtype = %s, want bf16", zb.DType())
	}
	for i, v := range zb.DataBF16() {
		if v != 0 {
			t.Errorf("ZerosBF16[%d] = %v, want 0", i, v)
		}
	}
}

func TestTensor_ToF32FromBF16(t *testing.T) {
	src := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	bf, _ := src.ToBF16()
	bfT := transformer.NewTensorFromBF16(bf, 4)
	f32, _ := bfT.ToF32()
	if len(f32) != 4 {
		t.Fatalf("ToF32 len = %d, want 4", len(f32))
	}
	for i, v := range f32 {
		if math.Abs(float64(v)-float64(i+1)) > 0.01 {
			t.Errorf("ToF32[%d] = %v, want %v", i, v, i+1)
		}
	}
}

func TestTensor_CloneIsDeepCopy(t *testing.T) {
	orig := transformer.NewTensor([]float32{1, 2, 3, 4}, 4)
	cl := orig.Clone()
	cl.DataF32()[0] = 99
	if orig.DataF32()[0] != 1 {
		t.Errorf("orig[0] = %v, want 1 (Clone must be deep)", orig.DataF32()[0])
	}
}

func TestTensor_Reshape(t *testing.T) {
	t1 := transformer.NewTensor([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	t2 := t1.Reshape(3, 2)
	if t2.Rank() != 2 || t2.Shape()[0] != 3 || t2.Shape()[1] != 2 {
		t.Errorf("Reshape(3,2) shape = %v, want [3 2]", t2.Shape())
	}
	if t2.Size() != 6 {
		t.Errorf("Reshape(3,2) size = %d, want 6", t2.Size())
	}
}

func TestTensor_ReshapePanicsOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on reshape size mismatch")
		}
	}()
	t1 := transformer.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	t1.Reshape(3, 2) // 6 elements, tensor has 4
}

func TestTensor_ShapeIsCopy(t *testing.T) {
	t1 := transformer.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
	shape := t1.Shape()
	shape[0] = 999
	if t1.Shape()[0] != 2 {
		t.Errorf("Shape mutation leaked: shape[0] = %d, want 2", t1.Shape()[0])
	}
}
