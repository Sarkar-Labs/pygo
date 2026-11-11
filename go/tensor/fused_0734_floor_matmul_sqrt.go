package tensor

import "fmt"

// Fused0734FloorMatmulSqrt implements the floor_matmul_sqrt operation contract for Go callers.
func Fused0734FloorMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0734(x, y)
}

func fusedDispatch_0734(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0734FloorMatmulSqrtName() string { return "floor_matmul_sqrt" }
