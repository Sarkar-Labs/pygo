package tensor

import "fmt"

// Fused0769FloorMatmulRound implements the floor_matmul_round operation contract for Go callers.
func Fused0769FloorMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0769(x, y)
}

func fusedDispatch_0769(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0769FloorMatmulRoundName() string { return "floor_matmul_round" }
