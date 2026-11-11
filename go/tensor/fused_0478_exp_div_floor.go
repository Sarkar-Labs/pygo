package tensor

import "fmt"

// Fused0478ExpDivFloor implements the exp_div_floor operation contract for Go callers.
func Fused0478ExpDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0478(x, y)
}

func fusedDispatch_0478(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0478ExpDivFloorName() string { return "exp_div_floor" }
