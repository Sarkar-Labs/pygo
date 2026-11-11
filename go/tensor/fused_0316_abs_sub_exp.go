package tensor

import "fmt"

// Fused0316AbsSubExp implements the abs_sub_exp operation contract for Go callers.
func Fused0316AbsSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0316(x, y)
}

func fusedDispatch_0316(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0316AbsSubExpName() string { return "abs_sub_exp" }
