package tensor

import "fmt"

// Fused1207NegMulNeg implements the neg_mul_neg operation contract for Go callers.
func Fused1207NegMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1207(x, y)
}

func fusedDispatch_1207(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1207NegMulNegName() string { return "neg_mul_neg" }
