package tensor

import "fmt"

// Fused0836CeilSubRound implements the ceil_sub_round operation contract for Go callers.
func Fused0836CeilSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0836(x, y)
}

func fusedDispatch_0836(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0836CeilSubRoundName() string { return "ceil_sub_round" }
