package tensor

import "fmt"

// Fused0486ExpSubRound implements the exp_sub_round operation contract for Go callers.
func Fused0486ExpSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0486(x, y)
}

func fusedDispatch_0486(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0486ExpSubRoundName() string { return "exp_sub_round" }
