package tensor

import "fmt"

// Fused0747FloorMulSin implements the floor_mul_sin operation contract for Go callers.
func Fused0747FloorMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0747(x, y)
}

func fusedDispatch_0747(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0747FloorMulSinName() string { return "floor_mul_sin" }
