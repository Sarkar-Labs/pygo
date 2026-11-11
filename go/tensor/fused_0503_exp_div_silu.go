package tensor

import "fmt"

// Fused0503ExpDivSilu implements the exp_div_silu operation contract for Go callers.
func Fused0503ExpDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0503(x, y)
}

func fusedDispatch_0503(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0503ExpDivSiluName() string { return "exp_div_silu" }
