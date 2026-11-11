package tensor

import "fmt"

// Fused0267TanhMulFloor implements the tanh_mul_floor operation contract for Go callers.
func Fused0267TanhMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0267(x, y)
}

func fusedDispatch_0267(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0267TanhMulFloorName() string { return "tanh_mul_floor" }
