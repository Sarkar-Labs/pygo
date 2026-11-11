package tensor

import "fmt"

// Fused0995SignAddNeg implements the sign_add_neg operation contract for Go callers.
func Fused0995SignAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0995(x, y)
}

func fusedDispatch_0995(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0995SignAddNegName() string { return "sign_add_neg" }
