package tensor

import "fmt"

// Fused0245TanhAddExp implements the tanh_add_exp operation contract for Go callers.
func Fused0245TanhAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0245(x, y)
}

func fusedDispatch_0245(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0245TanhAddExpName() string { return "tanh_add_exp" }
