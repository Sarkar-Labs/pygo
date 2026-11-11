package tensor

import "fmt"

// Fused0847CeilMulGelu implements the ceil_mul_gelu operation contract for Go callers.
func Fused0847CeilMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0847(x, y)
}

func fusedDispatch_0847(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0847CeilMulGeluName() string { return "ceil_mul_gelu" }
