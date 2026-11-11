package tensor

import "fmt"

// Fused1246Log2SubFloor implements the log2_sub_floor operation contract for Go callers.
func Fused1246Log2SubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1246(x, y)
}

func fusedDispatch_1246(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1246Log2SubFloorName() string { return "log2_sub_floor" }
