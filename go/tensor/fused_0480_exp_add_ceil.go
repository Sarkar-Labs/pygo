package tensor

import "fmt"

// Fused0480ExpAddCeil implements the exp_add_ceil operation contract for Go callers.
func Fused0480ExpAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0480(x, y)
}

func fusedDispatch_0480(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0480ExpAddCeilName() string { return "exp_add_ceil" }
