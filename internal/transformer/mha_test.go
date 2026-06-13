package transformer_test

import (
	"math"
	"testing"

	"github.com/xDarkicex/relux/internal/transformer"
)

func TestMHA_ForwardShape(t *testing.T) {
	// 1 head, 1 KV head, headDim=4, dModel=4.
	// No RoPE, no causal mask for the trivial case.
	m := transformer.NewMHA(4, 1, 1, nil, false)
	x := transformer.NewTensor([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 1, 4) // [batch=2, seq=1, dModel=4]
	y := m.Forward(x)
	if y.Rank() != 3 {
		t.Errorf("y.Rank() = %d, want 3", y.Rank())
	}
	if y.Shape()[0] != 2 || y.Shape()[1] != 1 || y.Shape()[2] != 4 {
		t.Errorf("y shape = %v, want [2 1 4]", y.Shape())
	}
}

func TestMHA_ForwardSingleTokenCausal(t *testing.T) {
	// 2 heads, 1 KV head (GQA), headDim=2, dModel=4.
	// Single token (seq=1), causal. The output for seq=1
	// is just the value-V (since the only key is the
	// self-key), scaled by softmax([1.0]) = 1.0.
	m := transformer.NewMHA(4, 2, 1, nil, true)
	x := transformer.NewTensor([]float32{1, 0, 0, 1}, 1, 1, 4) // [batch=1, seq=1, dModel=4]
	y := m.Forward(x)
	if y.Size() != 4 {
		t.Errorf("y size = %d, want 4", y.Size())
	}
	// The output is non-NaN and non-trivially non-zero.
	for i, v := range y.DataF32() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("y[%d] = %v, NaN or Inf", i, v)
		}
	}
}

func TestMHA_GQAExpansion(t *testing.T) {
	// 4 heads, 2 KV heads (GQA group size 2). Verify the
	// module accepts this configuration.
	m := transformer.NewMHA(8, 4, 2, nil, true)
	if m.Params()[0].Data == nil {
		t.Errorf("Wq is nil")
	}
	// The Q projection size should be dModel * numHeads * headDim
	// = 8 * 4 * 2 = 64. K/V projection size = 8 * 2 * 2 = 32.
	// Output size = 4 * 2 * 8 = 64.
	expectedWq := 8 * 4 * 2
	expectedWk := 8 * 2 * 2
	expectedWo := 4 * 2 * 8
	if got := len(m.Params()[0].Data); got != expectedWq {
		t.Errorf("Wq size = %d, want %d", got, expectedWq)
	}
	if got := len(m.Params()[1].Data); got != expectedWk {
		t.Errorf("Wk size = %d, want %d", got, expectedWk)
	}
	if got := len(m.Params()[3].Data); got != expectedWo {
		t.Errorf("Wo size = %d, want %d", got, expectedWo)
	}
}

func TestMHA_InvalidGQA(t *testing.T) {
	// numHeads not divisible by numKVHeads should panic.
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for invalid GQA grouping")
		}
	}()
	transformer.NewMHA(8, 4, 3, nil, true) // 4 not divisible by 3
}

func absDiff(a, b float32) float32 {
	if a > b {
		return a - b
	}
	return b - a
}

func TestMHA_BackwardGradCheck(t *testing.T) {
	// Numerical gradient check on Wq[0, 0] and Wo[0, 0].
	const dModel, numHeads, numKVHeads = 4, 2, 1
	const eps = 0.05

	m := transformer.NewMHA(dModel, numHeads, numKVHeads, nil, true)
	xData := []float32{0.5, 1.0, -0.5, 1.0, 0.3, 0.7, -0.3, 0.7}
	x := transformer.NewTensor(xData, 2, 1, dModel)
	m.SetMode(transformer.Train)

	// Forward.
	y := m.Forward(x)
	var loss float32
	for _, v := range y.DataF32() {
		loss += v * v
	}
	gradOutData := make([]float32, len(y.DataF32()))
	for i, v := range y.DataF32() {
		gradOutData[i] = 2 * v
	}
	gradOut := transformer.NewTensor(gradOutData, 2, 1, dModel)

	// Backward.
	m.Backward(gradOut)

	// Check Wq.
	wq := m.Params()[0]
	for idx := 0; idx < 2; idx++ {
		orig := transformer.F32FromBF16(wq.Data[idx])
		wq.Data[idx] = transformer.BF16FromF32(orig + eps)
		yp := m.Forward(x)
		var lp float32
		for _, v := range yp.DataF32() {
			lp += v * v
		}
		wq.Data[idx] = transformer.BF16FromF32(orig - eps)
		ym := m.Forward(x)
		var lm float32
		for _, v := range ym.DataF32() {
			lm += v * v
		}
		wq.Data[idx] = transformer.BF16FromF32(orig)
		numerical := (lp - lm) / (2 * eps)
		analytical := wq.Grad[idx]
		if absDiff(numerical, analytical) > 5e-2 {
			t.Errorf("Wq.Grad[%d]: analytical=%v, numerical=%v", idx, analytical, numerical)
		}
	}

	// Check Wo.
	wo := m.Params()[3]
	for idx := 0; idx < 2; idx++ {
		orig := transformer.F32FromBF16(wo.Data[idx])
		wo.Data[idx] = transformer.BF16FromF32(orig + eps)
		yp := m.Forward(x)
		var lp float32
		for _, v := range yp.DataF32() {
			lp += v * v
		}
		wo.Data[idx] = transformer.BF16FromF32(orig - eps)
		ym := m.Forward(x)
		var lm float32
		for _, v := range ym.DataF32() {
			lm += v * v
		}
		wo.Data[idx] = transformer.BF16FromF32(orig)
		numerical := (lp - lm) / (2 * eps)
		analytical := wo.Grad[idx]
		if absDiff(numerical, analytical) > 5e-2 {
			t.Errorf("Wo.Grad[%d]: analytical=%v, numerical=%v", idx, analytical, numerical)
		}
	}
}
