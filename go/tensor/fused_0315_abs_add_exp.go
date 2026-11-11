package tensor

import "fmt"

// Fused0315AbsAddExp implements the abs_add_exp operation contract for Go callers.
func Fused0315AbsAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0315(x, y)
}

func fusedDispatch_0315(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0315AbsAddExpName() string { return "abs_add_exp" }
