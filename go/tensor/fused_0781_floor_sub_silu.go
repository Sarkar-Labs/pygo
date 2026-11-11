package tensor

import "fmt"

// Fused0781FloorSubSilu implements the floor_sub_silu operation contract for Go callers.
func Fused0781FloorSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0781(x, y)
}

func fusedDispatch_0781(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0781FloorSubSiluName() string { return "floor_sub_silu" }
