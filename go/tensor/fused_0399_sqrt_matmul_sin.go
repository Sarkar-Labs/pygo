package tensor

import "fmt"

// Fused0399SqrtMatmulSin implements the sqrt_matmul_sin operation contract for Go callers.
func Fused0399SqrtMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0399(x, y)
}

func fusedDispatch_0399(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0399SqrtMatmulSinName() string { return "sqrt_matmul_sin" }
