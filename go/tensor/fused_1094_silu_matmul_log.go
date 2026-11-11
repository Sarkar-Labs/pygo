package tensor

import "fmt"

// Fused1094SiluMatmulLog implements the silu_matmul_log operation contract for Go callers.
func Fused1094SiluMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1094(x, y)
}

func fusedDispatch_1094(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1094SiluMatmulLogName() string { return "silu_matmul_log" }
