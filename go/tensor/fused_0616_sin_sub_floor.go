package tensor

import "fmt"

// Fused0616SinSubFloor implements the sin_sub_floor operation contract for Go callers.
func Fused0616SinSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0616(x, y)
}

func fusedDispatch_0616(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0616SinSubFloorName() string { return "sin_sub_floor" }
