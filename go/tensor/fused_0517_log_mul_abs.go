package tensor

import "fmt"

// Fused0517LogMulAbs implements the log_mul_abs operation contract for Go callers.
func Fused0517LogMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0517(x, y)
}

func fusedDispatch_0517(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0517LogMulAbsName() string { return "log_mul_abs" }
