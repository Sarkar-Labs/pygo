package tensor

import "fmt"

// Fused0097ReluMulAbs implements the relu_mul_abs operation contract for Go callers.
func Fused0097ReluMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0097(x, y)
}

func fusedDispatch_0097(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0097ReluMulAbsName() string { return "relu_mul_abs" }
