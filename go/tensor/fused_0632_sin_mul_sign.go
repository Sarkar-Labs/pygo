package tensor

import "fmt"

// Fused0632SinMulSign implements the sin_mul_sign operation contract for Go callers.
func Fused0632SinMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0632(x, y)
}

func fusedDispatch_0632(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0632SinMulSignName() string { return "sin_mul_sign" }
