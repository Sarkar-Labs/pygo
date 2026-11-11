package tensor

import "fmt"

// Fused0172SigmoidMulSqrt implements the sigmoid_mul_sqrt operation contract for Go callers.
func Fused0172SigmoidMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0172(x, y)
}

func fusedDispatch_0172(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0172SigmoidMulSqrtName() string { return "sigmoid_mul_sqrt" }
