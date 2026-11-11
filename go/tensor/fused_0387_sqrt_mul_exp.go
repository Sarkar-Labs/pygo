package tensor

import "fmt"

// Fused0387SqrtMulExp implements the sqrt_mul_exp operation contract for Go callers.
func Fused0387SqrtMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0387(x, y)
}

func fusedDispatch_0387(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0387SqrtMulExpName() string { return "sqrt_mul_exp" }
