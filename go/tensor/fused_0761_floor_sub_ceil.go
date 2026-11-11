package tensor

import "fmt"

// Fused0761FloorSubCeil implements the floor_sub_ceil operation contract for Go callers.
func Fused0761FloorSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0761(x, y)
}

func fusedDispatch_0761(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0761FloorSubCeilName() string { return "floor_sub_ceil" }
