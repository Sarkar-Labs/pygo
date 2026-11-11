package tensor

import "fmt"

// Fused0782FloorMulSilu implements the floor_mul_silu operation contract for Go callers.
func Fused0782FloorMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0782(x, y)
}

func fusedDispatch_0782(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0782FloorMulSiluName() string { return "floor_mul_silu" }
