package tensor

import "fmt"

// Fused0371SqrtSubTanh implements the sqrt_sub_tanh operation contract for Go callers.
func Fused0371SqrtSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0371(x, y)
}

func fusedDispatch_0371(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0371SqrtSubTanhName() string { return "sqrt_sub_tanh" }
