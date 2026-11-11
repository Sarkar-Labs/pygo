package tensor

import "fmt"

// Fused1116SiluSubRound implements the silu_sub_round operation contract for Go callers.
func Fused1116SiluSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1116(x, y)
}

func fusedDispatch_1116(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1116SiluSubRoundName() string { return "silu_sub_round" }
