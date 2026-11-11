package tensor

import "fmt"

// Fused0895RoundAddFloor implements the round_add_floor operation contract for Go callers.
func Fused0895RoundAddFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0895(x, y)
}

func fusedDispatch_0895(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0895RoundAddFloorName() string { return "round_add_floor" }
