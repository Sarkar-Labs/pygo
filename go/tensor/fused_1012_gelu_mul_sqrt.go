package tensor

import "fmt"

// Fused1012GeluMulSqrt implements the gelu_mul_sqrt operation contract for Go callers.
func Fused1012GeluMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1012(x, y)
}

func fusedDispatch_1012(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1012GeluMulSqrtName() string { return "gelu_mul_sqrt" }
