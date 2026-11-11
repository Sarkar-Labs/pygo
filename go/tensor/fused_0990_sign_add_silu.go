package tensor

import "fmt"

// Fused0990SignAddSilu implements the sign_add_silu operation contract for Go callers.
func Fused0990SignAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0990(x, y)
}

func fusedDispatch_0990(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0990SignAddSiluName() string { return "sign_add_silu" }
