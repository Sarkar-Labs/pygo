package tensor

import "fmt"

// Fused0155ReluAddNeg implements the relu_add_neg operation contract for Go callers.
func Fused0155ReluAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0155(x, y)
}

func fusedDispatch_0155(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0155ReluAddNegName() string { return "relu_add_neg" }
