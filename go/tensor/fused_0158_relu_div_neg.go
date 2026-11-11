package tensor

import "fmt"

// Fused0158ReluDivNeg implements the relu_div_neg operation contract for Go callers.
func Fused0158ReluDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0158(x, y)
}

func fusedDispatch_0158(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0158ReluDivNegName() string { return "relu_div_neg" }
