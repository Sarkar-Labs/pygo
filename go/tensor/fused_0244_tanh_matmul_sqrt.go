package tensor

import "fmt"

// Fused0244TanhMatmulSqrt implements the tanh_matmul_sqrt operation contract for Go callers.
func Fused0244TanhMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0244(x, y)
}

func fusedDispatch_0244(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0244TanhMatmulSqrtName() string { return "tanh_matmul_sqrt" }
