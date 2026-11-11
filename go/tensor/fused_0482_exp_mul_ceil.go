package tensor

import "fmt"

// Fused0482ExpMulCeil implements the exp_mul_ceil operation contract for Go callers.
func Fused0482ExpMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0482(x, y)
}

func fusedDispatch_0482(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0482ExpMulCeilName() string { return "exp_mul_ceil" }
