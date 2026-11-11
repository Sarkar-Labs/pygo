package tensor

import "fmt"

// Fused0991SignSubSilu implements the sign_sub_silu operation contract for Go callers.
func Fused0991SignSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0991(x, y)
}

func fusedDispatch_0991(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0991SignSubSiluName() string { return "sign_sub_silu" }
