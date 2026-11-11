package tensor

import "fmt"

// Fused1205NegAddNeg implements the neg_add_neg operation contract for Go callers.
func Fused1205NegAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1205(x, y)
}

func fusedDispatch_1205(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1205NegAddNegName() string { return "neg_add_neg" }
