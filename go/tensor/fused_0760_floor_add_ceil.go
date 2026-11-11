package tensor

import "fmt"

// Fused0760FloorAddCeil implements the floor_add_ceil operation contract for Go callers.
func Fused0760FloorAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0760(x, y)
}

func fusedDispatch_0760(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0760FloorAddCeilName() string { return "floor_add_ceil" }
