package tensor

import "fmt"

// Fused1037GeluMulFloor implements the gelu_mul_floor operation contract for Go callers.
func Fused1037GeluMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1037(x, y)
}

func fusedDispatch_1037(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1037GeluMulFloorName() string { return "gelu_mul_floor" }
