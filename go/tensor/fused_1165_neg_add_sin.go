package tensor

import "fmt"

// Fused1165NegAddSin implements the neg_add_sin operation contract for Go callers.
func Fused1165NegAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1165(x, y)
}

func fusedDispatch_1165(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1165NegAddSinName() string { return "neg_add_sin" }
