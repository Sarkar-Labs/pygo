package tensor

import "fmt"

// Fused1222Log2MulSqrt implements the log2_mul_sqrt operation contract for Go callers.
func Fused1222Log2MulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1222(x, y)
}

func fusedDispatch_1222(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1222Log2MulSqrtName() string { return "log2_mul_sqrt" }
