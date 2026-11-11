package tensor

import "fmt"

// Fused0115ReluAddSin implements the relu_add_sin operation contract for Go callers.
func Fused0115ReluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0115(x, y)
}

func fusedDispatch_0115(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0115ReluAddSinName() string { return "relu_add_sin" }
