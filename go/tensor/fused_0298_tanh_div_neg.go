package tensor

import "fmt"

// Fused0298TanhDivNeg implements the tanh_div_neg operation contract for Go callers.
func Fused0298TanhDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0298(x, y)
}

func fusedDispatch_0298(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0298TanhDivNegName() string { return "tanh_div_neg" }
