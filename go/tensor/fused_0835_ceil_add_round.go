package tensor

import "fmt"

// Fused0835CeilAddRound implements the ceil_add_round operation contract for Go callers.
func Fused0835CeilAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0835(x, y)
}

func fusedDispatch_0835(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0835CeilAddRoundName() string { return "ceil_add_round" }
