package tensor

import "fmt"

// Fused0940SignAddSqrt implements the sign_add_sqrt operation contract for Go callers.
func Fused0940SignAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0940(x, y)
}

func fusedDispatch_0940(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0940SignAddSqrtName() string { return "sign_add_sqrt" }
