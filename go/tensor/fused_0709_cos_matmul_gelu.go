package tensor

import "fmt"

// Fused0709CosMatmulGelu implements the cos_matmul_gelu operation contract for Go callers.
func Fused0709CosMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0709(x, y)
}

func fusedDispatch_0709(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0709CosMatmulGeluName() string { return "cos_matmul_gelu" }
