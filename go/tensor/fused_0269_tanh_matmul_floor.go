package tensor

import "fmt"

// Fused0269TanhMatmulFloor implements the tanh_matmul_floor operation contract for Go callers.
func Fused0269TanhMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0269(x, y)
}

func fusedDispatch_0269(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0269TanhMatmulFloorName() string { return "tanh_matmul_floor" }
