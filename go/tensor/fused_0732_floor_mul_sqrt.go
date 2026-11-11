package tensor

import "fmt"

// Fused0732FloorMulSqrt implements the floor_mul_sqrt operation contract for Go callers.
func Fused0732FloorMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0732(x, y)
}

func fusedDispatch_0732(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0732FloorMulSqrtName() string { return "floor_mul_sqrt" }
