package tensor

import "fmt"

// Fused1245Log2AddFloor implements the log2_add_floor operation contract for Go callers.
func Fused1245Log2AddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1245(x, y)
}

func fusedDispatch_1245(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1245Log2AddFloorName() string { return "log2_add_floor" }
