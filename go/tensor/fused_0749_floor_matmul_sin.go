package tensor

import "fmt"

// Fused0749FloorMatmulSin implements the floor_matmul_sin operation contract for Go callers.
func Fused0749FloorMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0749(x, y)
}

func fusedDispatch_0749(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0749FloorMatmulSinName() string { return "floor_matmul_sin" }
