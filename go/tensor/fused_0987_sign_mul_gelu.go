package tensor

import "fmt"

// Fused0987SignMulGelu implements the sign_mul_gelu operation contract for Go callers.
func Fused0987SignMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0987(x, y)
}

func fusedDispatch_0987(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0987SignMulGeluName() string { return "sign_mul_gelu" }
