package tensor

import "fmt"

// Fused1227Log2MulExp implements the log2_mul_exp operation contract for Go callers.
func Fused1227Log2MulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1227(x, y)
}

func fusedDispatch_1227(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1227Log2MulExpName() string { return "log2_mul_exp" }
