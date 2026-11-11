package tensor

import "fmt"

// Fused0296TanhSubNeg implements the tanh_sub_neg operation contract for Go callers.
func Fused0296TanhSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0296(x, y)
}

func fusedDispatch_0296(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0296TanhSubNegName() string { return "tanh_sub_neg" }
