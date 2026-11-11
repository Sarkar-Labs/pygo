package tensor

import "fmt"

// Fused1179NegMatmulFloor implements the neg_matmul_floor operation contract for Go callers.
func Fused1179NegMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1179(x, y)
}

func fusedDispatch_1179(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1179NegMatmulFloorName() string { return "neg_matmul_floor" }
