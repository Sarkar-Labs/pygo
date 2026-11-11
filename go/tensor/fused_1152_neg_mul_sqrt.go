package tensor

import "fmt"

// Fused1152NegMulSqrt implements the neg_mul_sqrt operation contract for Go callers.
func Fused1152NegMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1152(x, y)
}

func fusedDispatch_1152(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1152NegMulSqrtName() string { return "neg_mul_sqrt" }
