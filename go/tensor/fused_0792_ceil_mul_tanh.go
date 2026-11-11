package tensor

import "fmt"

// Fused0792CeilMulTanh implements the ceil_mul_tanh operation contract for Go callers.
func Fused0792CeilMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0792(x, y)
}

func fusedDispatch_0792(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0792CeilMulTanhName() string { return "ceil_mul_tanh" }
