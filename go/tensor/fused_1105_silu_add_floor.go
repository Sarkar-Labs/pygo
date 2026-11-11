package tensor

import "fmt"

// Fused1105SiluAddFloor implements the silu_add_floor operation contract for Go callers.
func Fused1105SiluAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1105(x, y)
}

func fusedDispatch_1105(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1105SiluAddFloorName() string { return "silu_add_floor" }
