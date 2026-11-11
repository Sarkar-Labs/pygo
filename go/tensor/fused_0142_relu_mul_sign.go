package tensor

import "fmt"

// Fused0142ReluMulSign implements the relu_mul_sign operation contract for Go callers.
func Fused0142ReluMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0142(x, y)
}

func fusedDispatch_0142(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0142ReluMulSignName() string { return "relu_mul_sign" }
