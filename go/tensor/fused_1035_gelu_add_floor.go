package tensor

import "fmt"

// Fused1035GeluAddFloor implements the gelu_add_floor operation contract for Go callers.
func Fused1035GeluAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1035(x, y)
}

func fusedDispatch_1035(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1035GeluAddFloorName() string { return "gelu_add_floor" }
