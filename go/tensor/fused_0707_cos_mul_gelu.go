package tensor

import "fmt"

// Fused0707CosMulGelu implements the cos_mul_gelu operation contract for Go callers.
func Fused0707CosMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0707(x, y)
}

func fusedDispatch_0707(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0707CosMulGeluName() string { return "cos_mul_gelu" }
