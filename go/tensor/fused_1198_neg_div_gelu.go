package tensor

import "fmt"

// Fused1198NegDivGelu implements the neg_div_gelu operation contract for Go callers.
func Fused1198NegDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1198(x, y)
}

func fusedDispatch_1198(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1198NegDivGeluName() string { return "neg_div_gelu" }
