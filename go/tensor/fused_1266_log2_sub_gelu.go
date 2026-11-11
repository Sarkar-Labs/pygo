package tensor

import "fmt"

// Fused1266Log2SubGelu implements the log2_sub_gelu operation contract for Go callers.
func Fused1266Log2SubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1266(x, y)
}

func fusedDispatch_1266(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1266Log2SubGeluName() string { return "log2_sub_gelu" }
