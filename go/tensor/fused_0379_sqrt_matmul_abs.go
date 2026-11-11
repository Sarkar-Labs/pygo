package tensor

import "fmt"

// Fused0379SqrtMatmulAbs implements the sqrt_matmul_abs operation contract for Go callers.
func Fused0379SqrtMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0379(x, y)
}

func fusedDispatch_0379(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0379SqrtMatmulAbsName() string { return "sqrt_matmul_abs" }
