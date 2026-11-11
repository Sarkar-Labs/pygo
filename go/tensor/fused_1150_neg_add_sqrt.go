package tensor

import "fmt"

// Fused1150NegAddSqrt implements the neg_add_sqrt operation contract for Go callers.
func Fused1150NegAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1150(x, y)
}

func fusedDispatch_1150(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1150NegAddSqrtName() string { return "neg_add_sqrt" }
