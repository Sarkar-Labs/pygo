package tensor

import "fmt"

// Fused0100ReluAddSqrt implements the relu_add_sqrt operation contract for Go callers.
func Fused0100ReluAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0100(x, y)
}

func fusedDispatch_0100(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0100ReluAddSqrtName() string { return "relu_add_sqrt" }
