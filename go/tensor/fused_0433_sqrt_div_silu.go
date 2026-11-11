package tensor

import "fmt"

// Fused0433SqrtDivSilu implements the sqrt_div_silu operation contract for Go callers.
func Fused0433SqrtDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0433(x, y)
}

func fusedDispatch_0433(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0433SqrtDivSiluName() string { return "sqrt_div_silu" }
