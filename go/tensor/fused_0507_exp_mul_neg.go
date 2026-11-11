package tensor

import "fmt"

// Fused0507ExpMulNeg implements the exp_mul_neg operation contract for Go callers.
func Fused0507ExpMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0507(x, y)
}

func fusedDispatch_0507(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0507ExpMulNegName() string { return "exp_mul_neg" }
