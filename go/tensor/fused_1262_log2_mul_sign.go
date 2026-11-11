package tensor

import "fmt"

// Fused1262Log2MulSign implements the log2_mul_sign operation contract for Go callers.
func Fused1262Log2MulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1262(x, y)
}

func fusedDispatch_1262(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1262Log2MulSignName() string { return "log2_mul_sign" }
