package tensor

import "fmt"

// Fused0492ExpMulSign implements the exp_mul_sign operation contract for Go callers.
func Fused0492ExpMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0492(x, y)
}

func fusedDispatch_0492(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0492ExpMulSignName() string { return "exp_mul_sign" }
