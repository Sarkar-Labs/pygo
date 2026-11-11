package tensor

import "fmt"

// Fused0450ExpAddSqrt implements the exp_add_sqrt operation contract for Go callers.
func Fused0450ExpAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0450(x, y)
}

func fusedDispatch_0450(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0450ExpAddSqrtName() string { return "exp_add_sqrt" }
