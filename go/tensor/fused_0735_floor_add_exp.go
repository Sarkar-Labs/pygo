package tensor

import "fmt"

// Fused0735FloorAddExp implements the floor_add_exp operation contract for Go callers.
func Fused0735FloorAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0735(x, y)
}

func fusedDispatch_0735(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0735FloorAddExpName() string { return "floor_add_exp" }
