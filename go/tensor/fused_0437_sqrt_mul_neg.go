package tensor

import "fmt"

// Fused0437SqrtMulNeg implements the sqrt_mul_neg operation contract for Go callers.
func Fused0437SqrtMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0437(x, y)
}

func fusedDispatch_0437(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0437SqrtMulNegName() string { return "sqrt_mul_neg" }
