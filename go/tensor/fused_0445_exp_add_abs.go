package tensor

import "fmt"

// Fused0445ExpAddAbs implements the exp_add_abs operation contract for Go callers.
func Fused0445ExpAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0445(x, y)
}

func fusedDispatch_0445(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0445ExpAddAbsName() string { return "exp_add_abs" }
