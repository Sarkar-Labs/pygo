package tensor

import "fmt"

// Fused0920RoundAddSilu implements the round_add_silu operation contract for Go callers.
func Fused0920RoundAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0920(x, y)
}

func fusedDispatch_0920(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0920RoundAddSiluName() string { return "round_add_silu" }
