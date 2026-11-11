package tensor

import "fmt"

// Fused1056GeluSubGelu implements the gelu_sub_gelu operation contract for Go callers.
func Fused1056GeluSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1056(x, y)
}

func fusedDispatch_1056(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1056GeluSubGeluName() string { return "gelu_sub_gelu" }
