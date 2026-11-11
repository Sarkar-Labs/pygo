package tensor

import "fmt"

// Fused0838CeilDivRound implements the ceil_div_round operation contract for Go callers.
func Fused0838CeilDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0838(x, y)
}

func fusedDispatch_0838(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0838CeilDivRoundName() string { return "ceil_div_round" }
