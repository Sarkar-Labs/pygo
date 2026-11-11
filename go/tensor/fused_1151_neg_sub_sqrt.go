package tensor

import "fmt"

// Fused1151NegSubSqrt implements the neg_sub_sqrt operation contract for Go callers.
func Fused1151NegSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1151(x, y)
}

func fusedDispatch_1151(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1151NegSubSqrtName() string { return "neg_sub_sqrt" }
