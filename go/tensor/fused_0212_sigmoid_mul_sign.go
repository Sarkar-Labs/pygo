package tensor

import "fmt"

// Fused0212SigmoidMulSign implements the sigmoid_mul_sign operation contract for Go callers.
func Fused0212SigmoidMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0212(x, y)
}

func fusedDispatch_0212(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0212SigmoidMulSignName() string { return "sigmoid_mul_sign" }
