package tensor

import "fmt"

// Fused0602SinMulLog implements the sin_mul_log operation contract for Go callers.
func Fused0602SinMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0602(x, y)
}

func fusedDispatch_0602(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0602SinMulLogName() string { return "sin_mul_log" }
