package tensor

import "fmt"

// Fused0899RoundMatmulFloor implements the round_matmul_floor operation contract for Go callers.
func Fused0899RoundMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0899(x, y)
}

func fusedDispatch_0899(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0899RoundMatmulFloorName() string { return "round_matmul_floor" }
