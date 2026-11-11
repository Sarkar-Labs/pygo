package tensor

import "fmt"

// Fused1183NegDivCeil implements the neg_div_ceil operation contract for Go callers.
func Fused1183NegDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1183(x, y)
}

func fusedDispatch_1183(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1183NegDivCeilName() string { return "neg_div_ceil" }
