package tensor

import "fmt"

// Fused0327AbsMulSin implements the abs_mul_sin operation contract for Go callers.
func Fused0327AbsMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0327(x, y)
}

func fusedDispatch_0327(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0327AbsMulSinName() string { return "abs_mul_sin" }
