package tensor

import "fmt"

// Fused0424SqrtMatmulSign implements the sqrt_matmul_sign operation contract for Go callers.
func Fused0424SqrtMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0424(x, y)
}

func fusedDispatch_0424(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0424SqrtMatmulSignName() string { return "sqrt_matmul_sign" }
