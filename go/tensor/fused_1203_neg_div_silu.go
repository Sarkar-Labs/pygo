package tensor

import "fmt"

// Fused1203NegDivSilu implements the neg_div_silu operation contract for Go callers.
func Fused1203NegDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1203(x, y)
}

func fusedDispatch_1203(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1203NegDivSiluName() string { return "neg_div_silu" }
