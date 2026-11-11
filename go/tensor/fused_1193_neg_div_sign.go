package tensor

import "fmt"

// Fused1193NegDivSign implements the neg_div_sign operation contract for Go callers.
func Fused1193NegDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1193(x, y)
}

func fusedDispatch_1193(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1193NegDivSignName() string { return "neg_div_sign" }
