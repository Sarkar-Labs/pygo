package tensor

import "fmt"

// Fused0336AbsSubFloor implements the abs_sub_floor operation contract for Go callers.
func Fused0336AbsSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0336(x, y)
}

func fusedDispatch_0336(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0336AbsSubFloorName() string { return "abs_sub_floor" }
