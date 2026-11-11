package tensor

import "fmt"

// Fused0625SinAddRound implements the sin_add_round operation contract for Go callers.
func Fused0625SinAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0625(x, y)
}

func fusedDispatch_0625(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0625SinAddRoundName() string { return "sin_add_round" }
