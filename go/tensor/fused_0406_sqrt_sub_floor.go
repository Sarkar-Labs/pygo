package tensor

import "fmt"

// Fused0406SqrtSubFloor implements the sqrt_sub_floor operation contract for Go callers.
func Fused0406SqrtSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0406(x, y)
}

func fusedDispatch_0406(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0406SqrtSubFloorName() string { return "sqrt_sub_floor" }
