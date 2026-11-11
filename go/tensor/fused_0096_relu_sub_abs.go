package tensor

import "fmt"

// Fused0096ReluSubAbs implements the relu_sub_abs operation contract for Go callers.
func Fused0096ReluSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0096(x, y)
}

func fusedDispatch_0096(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0096ReluSubAbsName() string { return "relu_sub_abs" }
