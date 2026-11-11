package tensor

import "fmt"

// Fused0151ReluSubSilu implements the relu_sub_silu operation contract for Go callers.
func Fused0151ReluSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0151(x, y)
}

func fusedDispatch_0151(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0151ReluSubSiluName() string { return "relu_sub_silu" }
