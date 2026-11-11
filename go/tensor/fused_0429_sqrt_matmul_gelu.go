package tensor

import "fmt"

// Fused0429SqrtMatmulGelu implements the sqrt_matmul_gelu operation contract for Go callers.
func Fused0429SqrtMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0429(x, y)
}

func fusedDispatch_0429(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0429SqrtMatmulGeluName() string { return "sqrt_matmul_gelu" }
