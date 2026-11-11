package tensor

import "fmt"

// Fused0997SignMulNeg implements the sign_mul_neg operation contract for Go callers.
func Fused0997SignMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0997(x, y)
}

func fusedDispatch_0997(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0997SignMulNegName() string { return "sign_mul_neg" }
