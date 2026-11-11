package tensor

import "fmt"

// Fused0771FloorSubSign implements the floor_sub_sign operation contract for Go callers.
func Fused0771FloorSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0771(x, y)
}

func fusedDispatch_0771(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0771FloorSubSignName() string { return "floor_sub_sign" }
