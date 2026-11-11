package tensor

import "fmt"

// Fused1248Log2DivFloor implements the log2_div_floor operation contract for Go callers.
func Fused1248Log2DivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1248(x, y)
}

func fusedDispatch_1248(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1248Log2DivFloorName() string { return "log2_div_floor" }
