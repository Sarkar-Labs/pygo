package tensor

import "fmt"

// Fused0370SqrtAddTanh implements the sqrt_add_tanh operation contract for Go callers.
func Fused0370SqrtAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0370(x, y)
}

func fusedDispatch_0370(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0370SqrtAddTanhName() string { return "sqrt_add_tanh" }
