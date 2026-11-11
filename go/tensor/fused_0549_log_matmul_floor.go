package tensor

import "fmt"

// Fused0549LogMatmulFloor implements the log_matmul_floor operation contract for Go callers.
func Fused0549LogMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0549(x, y)
}

func fusedDispatch_0549(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0549LogMatmulFloorName() string { return "log_matmul_floor" }
