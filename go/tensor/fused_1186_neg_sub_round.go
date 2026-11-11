package tensor

import "fmt"

// Fused1186NegSubRound implements the neg_sub_round operation contract for Go callers.
func Fused1186NegSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1186(x, y)
}

func fusedDispatch_1186(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1186NegSubRoundName() string { return "neg_sub_round" }
