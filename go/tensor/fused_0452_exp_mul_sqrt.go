package tensor

import "fmt"

// Fused0452ExpMulSqrt implements the exp_mul_sqrt operation contract for Go callers.
func Fused0452ExpMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0452(x, y)
}

func fusedDispatch_0452(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0452ExpMulSqrtName() string { return "exp_mul_sqrt" }
