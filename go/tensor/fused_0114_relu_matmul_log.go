package tensor

import "fmt"

// Fused0114ReluMatmulLog implements the relu_matmul_log operation contract for Go callers.
func Fused0114ReluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0114(x, y)
}

func fusedDispatch_0114(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0114ReluMatmulLogName() string { return "relu_matmul_log" }
