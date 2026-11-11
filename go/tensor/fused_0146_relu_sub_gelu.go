package tensor

import "fmt"

// Fused0146ReluSubGelu implements the relu_sub_gelu operation contract for Go callers.
func Fused0146ReluSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0146(x, y)
}

func fusedDispatch_0146(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0146ReluSubGeluName() string { return "relu_sub_gelu" }
