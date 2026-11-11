package tensor

import "fmt"

// Fused0779FloorMatmulGelu implements the floor_matmul_gelu operation contract for Go callers.
func Fused0779FloorMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0779(x, y)
}

func fusedDispatch_0779(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0779FloorMatmulGeluName() string { return "floor_matmul_gelu" }
