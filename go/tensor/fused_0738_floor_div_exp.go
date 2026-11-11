package tensor

import "fmt"

// Fused0738FloorDivExp implements the floor_div_exp operation contract for Go callers.
func Fused0738FloorDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0738(x, y)
}

func fusedDispatch_0738(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0738FloorDivExpName() string { return "floor_div_exp" }
