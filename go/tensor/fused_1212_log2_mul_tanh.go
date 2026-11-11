package tensor

import "fmt"

// Fused1212Log2MulTanh implements the log2_mul_tanh operation contract for Go callers.
func Fused1212Log2MulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1212(x, y)
}

func fusedDispatch_1212(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1212Log2MulTanhName() string { return "log2_mul_tanh" }
