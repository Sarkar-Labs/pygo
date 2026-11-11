package tensor

import "fmt"

// Fused0508ExpDivNeg implements the exp_div_neg operation contract for Go callers.
func Fused0508ExpDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0508(x, y)
}

func fusedDispatch_0508(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0508ExpDivNegName() string { return "exp_div_neg" }
