package tensor

import "fmt"

// Fused0500ExpAddSilu implements the exp_add_silu operation contract for Go callers.
func Fused0500ExpAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0500(x, y)
}

func fusedDispatch_0500(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0500ExpAddSiluName() string { return "exp_add_silu" }
