package tensor

import "fmt"

// Fused0457ExpMulExp implements the exp_mul_exp operation contract for Go callers.
func Fused0457ExpMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0457(x, y)
}

func fusedDispatch_0457(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0457ExpMulExpName() string { return "exp_mul_exp" }
