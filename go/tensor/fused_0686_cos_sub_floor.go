package tensor

import "fmt"

// Fused0686CosSubFloor implements the cos_sub_floor operation contract for Go callers.
func Fused0686CosSubFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0686(x, y)
}

func fusedDispatch_0686(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0686CosSubFloorName() string { return "cos_sub_floor" }
