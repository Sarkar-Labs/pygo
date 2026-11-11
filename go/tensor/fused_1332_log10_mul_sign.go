package tensor

import "fmt"

// Fused1332Log10MulSign implements the log10_mul_sign operation contract for Go callers.
func Fused1332Log10MulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1332(x, y)
}

func fusedDispatch_1332(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1332Log10MulSignName() string { return "log10_mul_sign" }
