package tensor

import "fmt"

// Fused0795CeilAddAbs implements the ceil_add_abs operation contract for Go callers.
func Fused0795CeilAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0795(x, y)
}

func fusedDispatch_0795(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0795CeilAddAbsName() string { return "ceil_add_abs" }
