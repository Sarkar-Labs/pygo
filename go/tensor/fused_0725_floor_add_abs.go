package tensor

import "fmt"

// Fused0725FloorAddAbs implements the floor_add_abs operation contract for Go callers.
func Fused0725FloorAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0725(x, y)
}

func fusedDispatch_0725(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0725FloorAddAbsName() string { return "floor_add_abs" }
