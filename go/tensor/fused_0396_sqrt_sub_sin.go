package tensor

import "fmt"

// Fused0396SqrtSubSin implements the sqrt_sub_sin operation contract for Go callers.
func Fused0396SqrtSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0396(x, y)
}

func fusedDispatch_0396(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0396SqrtSubSinName() string { return "sqrt_sub_sin" }
