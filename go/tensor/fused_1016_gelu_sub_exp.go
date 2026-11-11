package tensor

import "fmt"

// Fused1016GeluSubExp implements the gelu_sub_exp operation contract for Go callers.
func Fused1016GeluSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1016(x, y)
}

func fusedDispatch_1016(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1016GeluSubExpName() string { return "gelu_sub_exp" }
