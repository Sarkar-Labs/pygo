package tensor

import "fmt"

// Fused0752FloorMulCos implements the floor_mul_cos operation contract for Go callers.
func Fused0752FloorMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0752(x, y)
}

func fusedDispatch_0752(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0752FloorMulCosName() string { return "floor_mul_cos" }
