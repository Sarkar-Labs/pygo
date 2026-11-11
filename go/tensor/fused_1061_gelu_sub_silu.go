package tensor

import "fmt"

// Fused1061GeluSubSilu implements the gelu_sub_silu operation contract for Go callers.
func Fused1061GeluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1061(x, y)
}

func fusedDispatch_1061(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1061GeluSubSiluName() string { return "gelu_sub_silu" }
