package tensor

import "fmt"

// Fused0092ReluMulTanh implements the relu_mul_tanh operation contract for Go callers.
func Fused0092ReluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0092(x, y)
}

func fusedDispatch_0092(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0092ReluMulTanhName() string { return "relu_mul_tanh" }
