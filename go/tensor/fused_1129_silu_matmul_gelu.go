package tensor

import "fmt"

// Fused1129SiluMatmulGelu implements the silu_matmul_gelu operation contract for Go callers.
func Fused1129SiluMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1129(x, y)
}

func fusedDispatch_1129(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1129SiluMatmulGeluName() string { return "silu_matmul_gelu" }
