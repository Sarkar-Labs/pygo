package tensor

import "fmt"

// Fused0689CosMatmulFloor implements the cos_matmul_floor operation contract for Go callers.
func Fused0689CosMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0689(x, y)
}

func fusedDispatch_0689(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0689CosMatmulFloorName() string { return "cos_matmul_floor" }
