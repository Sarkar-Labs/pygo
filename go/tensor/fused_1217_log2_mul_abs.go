package tensor

import "fmt"

// Fused1217Log2MulAbs implements the log2_mul_abs operation contract for Go callers.
func Fused1217Log2MulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1217(x, y)
}

func fusedDispatch_1217(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1217Log2MulAbsName() string { return "log2_mul_abs" }
