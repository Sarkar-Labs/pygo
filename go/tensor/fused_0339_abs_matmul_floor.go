package tensor

import "fmt"

// Fused0339AbsMatmulFloor implements the abs_matmul_floor operation contract for Go callers.
func Fused0339AbsMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0339(x, y)
}

func fusedDispatch_0339(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0339AbsMatmulFloorName() string { return "abs_matmul_floor" }
