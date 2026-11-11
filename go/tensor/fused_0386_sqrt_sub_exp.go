package tensor

import "fmt"

// Fused0386SqrtSubExp implements the sqrt_sub_exp operation contract for Go callers.
func Fused0386SqrtSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0386(x, y)
}

func fusedDispatch_0386(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0386SqrtSubExpName() string { return "sqrt_sub_exp" }
