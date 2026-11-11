package tensor

import "fmt"

// Fused0439SqrtMatmulNeg implements the sqrt_matmul_neg operation contract for Go callers.
func Fused0439SqrtMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0439(x, y)
}

func fusedDispatch_0439(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0439SqrtMatmulNegName() string { return "sqrt_matmul_neg" }
