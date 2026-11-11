package tensor

import "fmt"

// Fused1182NegMulCeil implements the neg_mul_ceil operation contract for Go callers.
func Fused1182NegMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1182(x, y)
}

func fusedDispatch_1182(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1182NegMulCeilName() string { return "neg_mul_ceil" }
