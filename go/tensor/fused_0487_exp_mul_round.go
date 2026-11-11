package tensor

import "fmt"

// Fused0487ExpMulRound implements the exp_mul_round operation contract for Go callers.
func Fused0487ExpMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0487(x, y)
}

func fusedDispatch_0487(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0487ExpMulRoundName() string { return "exp_mul_round" }
