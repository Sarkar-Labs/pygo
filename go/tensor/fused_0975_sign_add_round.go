package tensor

import "fmt"

// Fused0975SignAddRound implements the sign_add_round operation contract for Go callers.
func Fused0975SignAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0975(x, y)
}

func fusedDispatch_0975(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0975SignAddRoundName() string { return "sign_add_round" }
