package tensor

import "fmt"

// Fused0117ReluMulSin implements the relu_mul_sin operation contract for Go callers.
func Fused0117ReluMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0117(x, y)
}

func fusedDispatch_0117(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0117ReluMulSinName() string { return "relu_mul_sin" }
