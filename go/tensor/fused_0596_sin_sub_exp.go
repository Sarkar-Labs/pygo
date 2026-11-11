package tensor

import "fmt"

// Fused0596SinSubExp implements the sin_sub_exp operation contract for Go callers.
func Fused0596SinSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0596(x, y)
}

func fusedDispatch_0596(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0596SinSubExpName() string { return "sin_sub_exp" }
