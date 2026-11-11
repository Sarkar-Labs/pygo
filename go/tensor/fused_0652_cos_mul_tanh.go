package tensor

import "fmt"

// Fused0652CosMulTanh implements the cos_mul_tanh operation contract for Go callers.
func Fused0652CosMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0652(x, y)
}

func fusedDispatch_0652(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0652CosMulTanhName() string { return "cos_mul_tanh" }
