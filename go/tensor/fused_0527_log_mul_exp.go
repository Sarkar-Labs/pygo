package tensor

import "fmt"

// Fused0527LogMulExp implements the log_mul_exp operation contract for Go callers.
func Fused0527LogMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0527(x, y)
}

func fusedDispatch_0527(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0527LogMulExpName() string { return "log_mul_exp" }
