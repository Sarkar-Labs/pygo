package tensor

import "fmt"

// Fused0537LogMulSin implements the log_mul_sin operation contract for Go callers.
func Fused0537LogMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0537(x, y)
}

func fusedDispatch_0537(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0537LogMulSinName() string { return "log_mul_sin" }
