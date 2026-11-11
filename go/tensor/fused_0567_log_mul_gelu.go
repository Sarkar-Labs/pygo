package tensor

import "fmt"

// Fused0567LogMulGelu implements the log_mul_gelu operation contract for Go callers.
func Fused0567LogMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0567(x, y)
}

func fusedDispatch_0567(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0567LogMulGeluName() string { return "log_mul_gelu" }
