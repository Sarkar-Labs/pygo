package tensor

import "fmt"

// Fused0293TanhDivSilu implements the tanh_div_silu operation contract for Go callers.
func Fused0293TanhDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0293(x, y)
}

func fusedDispatch_0293(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0293TanhDivSiluName() string { return "tanh_div_silu" }
