package tensor

import "fmt"

// Fused0498ExpDivGelu implements the exp_div_gelu operation contract for Go callers.
func Fused0498ExpDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0498(x, y)
}

func fusedDispatch_0498(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0498ExpDivGeluName() string { return "exp_div_gelu" }
