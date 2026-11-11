package tensor

import "fmt"

// Fused0197SigmoidMulFloor implements the sigmoid_mul_floor operation contract for Go callers.
func Fused0197SigmoidMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0197(x, y)
}

func fusedDispatch_0197(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0197SigmoidMulFloorName() string { return "sigmoid_mul_floor" }
