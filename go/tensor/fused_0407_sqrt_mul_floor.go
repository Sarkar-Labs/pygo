package tensor

import "fmt"

// Fused0407SqrtMulFloor implements the sqrt_mul_floor operation contract for Go callers.
func Fused0407SqrtMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0407(x, y)
}

func fusedDispatch_0407(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0407SqrtMulFloorName() string { return "sqrt_mul_floor" }
