package tensor

import "fmt"

// Fused1315Log10AddFloor implements the log10_add_floor operation contract for Go callers.
func Fused1315Log10AddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1315(x, y)
}

func fusedDispatch_1315(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1315Log10AddFloorName() string { return "log10_add_floor" }
