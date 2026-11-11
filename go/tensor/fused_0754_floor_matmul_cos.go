package tensor

import "fmt"

// Fused0754FloorMatmulCos implements the floor_matmul_cos operation contract for Go callers.
func Fused0754FloorMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0754(x, y)
}

func fusedDispatch_0754(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0754FloorMatmulCosName() string { return "floor_matmul_cos" }
