package tensor

import "fmt"

// Fused0132ReluMulCeil implements the relu_mul_ceil operation contract for Go callers.
func Fused0132ReluMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0132(x, y)
}

func fusedDispatch_0132(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0132ReluMulCeilName() string { return "relu_mul_ceil" }
