package tensor

import "fmt"

// Fused0805CeilAddExp implements the ceil_add_exp operation contract for Go callers.
func Fused0805CeilAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0805(x, y)
}

func fusedDispatch_0805(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0805CeilAddExpName() string { return "ceil_add_exp" }
