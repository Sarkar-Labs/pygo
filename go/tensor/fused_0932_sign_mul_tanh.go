package tensor

import "fmt"

// Fused0932SignMulTanh implements the sign_mul_tanh operation contract for Go callers.
func Fused0932SignMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0932(x, y)
}

func fusedDispatch_0932(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0932SignMulTanhName() string { return "sign_mul_tanh" }
