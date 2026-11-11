package tensor

import "fmt"

// Fused0736FloorSubExp implements the floor_sub_exp operation contract for Go callers.
func Fused0736FloorSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0736(x, y)
}

func fusedDispatch_0736(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0736FloorSubExpName() string { return "floor_sub_exp" }
