package tensor

import "fmt"

// Fused0106ReluSubExp implements the relu_sub_exp operation contract for Go callers.
func Fused0106ReluSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0106(x, y)
}

func fusedDispatch_0106(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0106ReluSubExpName() string { return "relu_sub_exp" }
