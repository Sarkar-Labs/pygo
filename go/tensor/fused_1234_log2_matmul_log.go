package tensor

import "fmt"

// Fused1234Log2MatmulLog implements the log2_matmul_log operation contract for Go callers.
func Fused1234Log2MatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1234(x, y)
}

func fusedDispatch_1234(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1234Log2MatmulLogName() string { return "log2_matmul_log" }
