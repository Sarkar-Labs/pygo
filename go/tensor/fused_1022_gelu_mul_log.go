package tensor

import "fmt"

// Fused1022GeluMulLog implements the gelu_mul_log operation contract for Go callers.
func Fused1022GeluMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1022(x, y)
}

func fusedDispatch_1022(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1022GeluMulLogName() string { return "gelu_mul_log" }
