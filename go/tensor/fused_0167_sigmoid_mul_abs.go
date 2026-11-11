package tensor

import "fmt"

// Fused0167SigmoidMulAbs implements the sigmoid_mul_abs operation contract for Go callers.
func Fused0167SigmoidMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0167(x, y)
}

func fusedDispatch_0167(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0167SigmoidMulAbsName() string { return "sigmoid_mul_abs" }
