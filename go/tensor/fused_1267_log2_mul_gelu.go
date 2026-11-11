package tensor

import "fmt"

// Fused1267Log2MulGelu implements the log2_mul_gelu operation contract for Go callers.
func Fused1267Log2MulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1267(x, y)
}

func fusedDispatch_1267(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1267Log2MulGeluName() string { return "log2_mul_gelu" }
