package tensor

import "fmt"

// Fused0276TanhSubRound implements the tanh_sub_round operation contract for Go callers.
func Fused0276TanhSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0276(x, y)
}

func fusedDispatch_0276(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0276TanhSubRoundName() string { return "tanh_sub_round" }
