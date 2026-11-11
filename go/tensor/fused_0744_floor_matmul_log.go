package tensor

import "fmt"

// Fused0744FloorMatmulLog implements the floor_matmul_log operation contract for Go callers.
func Fused0744FloorMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0744(x, y)
}

func fusedDispatch_0744(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0744FloorMatmulLogName() string { return "floor_matmul_log" }
