package transformer

// matmulFloat32 computes C = A @ B where A is [M, K], B is
// [K, N], C is [M, N]. Row-major; all slices have length
// M*K, K*N, M*N respectively. Pure Go, no BLAS.
//
// The inner loop is the K reduction; the Go compiler can
// auto-vectorize the inner-K loop when it's the tightest
// one (which it is for typical dModel sizes up to a few
// thousand). The follow-up routes this through rnxa's
// float32 GEMM for the GPU path.
//
// out is required to be a pre-allocated slice of length M*N.
// The caller is responsible for its lifetime.
func matmulFloat32(out, a, b []float32, M, K, N int) {
	for i := 0; i < M; i++ {
		aRow := i * K
		outRow := i * N
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a[aRow+k] * b[k*N+j]
			}
			out[outRow+j] = sum
		}
	}
}

// matmulFloat32TransB computes C = A @ B^T where A is
// [M, K], B is [N, K] (B^T is [K, N]), C is [M, N]. Same
// row-major convention.
func matmulFloat32TransB(out, a, b []float32, M, K, N int) {
	for i := 0; i < M; i++ {
		aRow := i * K
		outRow := i * N
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a[aRow+k] * b[j*K+k]
			}
			out[outRow+j] = sum
		}
	}
}

// transpose4D swaps the last two axes of a 4D tensor
// [a, b, c, d] -> [a, b, d, c]. The output is a fresh
// allocation (caller owns).
func transpose4D(src []float32, shape []int) []float32 {
	if len(shape) != 4 {
		panic("transpose4D: shape must be 4D")
	}
	A, B, C, D := shape[0], shape[1], shape[2], shape[3]
	out := make([]float32, A*B*C*D)
	rowSize := C * D
	for a := 0; a < A; a++ {
		for b := 0; b < B; b++ {
			for c := 0; c < C; c++ {
				for d := 0; d < D; d++ {
					srcIdx := ((a*B+b)*C+c)*D + d
					dstIdx := ((a*B+b)*D+d)*C + c
					out[dstIdx] = src[srcIdx]
				}
			}
		}
	}
	_ = rowSize
	return out
}
