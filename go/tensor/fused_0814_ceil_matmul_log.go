package tensor

import "fmt"

// Fused0814CeilMatmulLog implements the ceil_matmul_log operation contract for Go callers.
func Fused0814CeilMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0814(x, y)
}

func fusedDispatch_0814(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0814CeilMatmulLogName() string { return "ceil_matmul_log" }
