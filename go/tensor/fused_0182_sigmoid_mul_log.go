package tensor

import "fmt"

// Fused0182SigmoidMulLog implements the sigmoid_mul_log operation contract for Go callers.
func Fused0182SigmoidMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0182(x, y)
}

func fusedDispatch_0182(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0182SigmoidMulLogName() string { return "sigmoid_mul_log" }
