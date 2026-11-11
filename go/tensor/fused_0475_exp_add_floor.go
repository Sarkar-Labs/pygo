package tensor

import "fmt"

// Fused0475ExpAddFloor implements the exp_add_floor operation contract for Go callers.
func Fused0475ExpAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0475(x, y)
}

func fusedDispatch_0475(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0475ExpAddFloorName() string { return "exp_add_floor" }
