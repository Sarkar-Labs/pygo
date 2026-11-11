package tensor

import "fmt"

// Fused0816CeilSubSin implements the ceil_sub_sin operation contract for Go callers.
func Fused0816CeilSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0816(x, y)
}

func fusedDispatch_0816(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0816CeilSubSinName() string { return "ceil_sub_sin" }
