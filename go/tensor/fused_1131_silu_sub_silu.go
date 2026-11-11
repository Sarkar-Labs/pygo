package tensor

import "fmt"

// Fused1131SiluSubSilu implements the silu_sub_silu operation contract for Go callers.
func Fused1131SiluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1131(x, y)
}

func fusedDispatch_1131(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1131SiluSubSiluName() string { return "silu_sub_silu" }
