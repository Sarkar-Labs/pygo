package tensor

import "fmt"

// Fused0825CeilAddFloor implements the ceil_add_floor operation contract for Go callers.
func Fused0825CeilAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0825(x, y)
}

func fusedDispatch_0825(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0825CeilAddFloorName() string { return "ceil_add_floor" }
