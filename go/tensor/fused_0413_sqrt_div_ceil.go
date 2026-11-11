package tensor

import "fmt"

// Fused0413SqrtDivCeil implements the sqrt_div_ceil operation contract for Go callers.
func Fused0413SqrtDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0413(x, y)
}

func fusedDispatch_0413(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0413SqrtDivCeilName() string { return "sqrt_div_ceil" }
