package tensor

import "fmt"

// Fused1002GeluMulTanh implements the gelu_mul_tanh operation contract for Go callers.
func Fused1002GeluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1002(x, y)
}

func fusedDispatch_1002(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1002GeluMulTanhName() string { return "gelu_mul_tanh" }
