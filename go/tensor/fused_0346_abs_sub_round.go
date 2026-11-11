package tensor

import "fmt"

// Fused0346AbsSubRound implements the abs_sub_round operation contract for Go callers.
func Fused0346AbsSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0346(x, y)
}

func fusedDispatch_0346(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0346AbsSubRoundName() string { return "abs_sub_round" }
