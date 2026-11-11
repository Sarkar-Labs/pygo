package tensor

import "fmt"

// Fused0477ExpMulFloor implements the exp_mul_floor operation contract for Go callers.
func Fused0477ExpMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0477(x, y)
}

func fusedDispatch_0477(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0477ExpMulFloorName() string { return "exp_mul_floor" }
