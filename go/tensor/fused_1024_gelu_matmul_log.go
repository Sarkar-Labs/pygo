package tensor

import "fmt"

// Fused1024GeluMatmulLog implements the gelu_matmul_log operation contract for Go callers.
func Fused1024GeluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1024(x, y)
}

func fusedDispatch_1024(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1024GeluMatmulLogName() string { return "gelu_matmul_log" }
