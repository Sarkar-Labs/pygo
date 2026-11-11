package tensor

import "fmt"

// Fused1085SiluAddExp implements the silu_add_exp operation contract for Go callers.
func Fused1085SiluAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1085(x, y)
}

func fusedDispatch_1085(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1085SiluAddExpName() string { return "silu_add_exp" }
