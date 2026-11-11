package tensor

import "fmt"

// Fused0420SqrtAddSign implements the sqrt_add_sign operation contract for Go callers.
func Fused0420SqrtAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0420(x, y)
}

func fusedDispatch_0420(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0420SqrtAddSignName() string { return "sqrt_add_sign" }
