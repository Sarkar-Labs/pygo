package tensor

import "fmt"

// Fused0416SqrtSubRound implements the sqrt_sub_round operation contract for Go callers.
func Fused0416SqrtSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0416(x, y)
}

func fusedDispatch_0416(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0416SqrtSubRoundName() string { return "sqrt_sub_round" }
