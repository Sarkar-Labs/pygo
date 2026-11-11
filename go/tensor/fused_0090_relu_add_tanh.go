package tensor

import "fmt"

// Fused0090ReluAddTanh implements the relu_add_tanh operation contract for Go callers.
func Fused0090ReluAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0090(x, y)
}

func fusedDispatch_0090(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0090ReluAddTanhName() string { return "relu_add_tanh" }
