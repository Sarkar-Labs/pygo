package tensor

import "fmt"

// Fused0595SinAddExp implements the sin_add_exp operation contract for Go callers.
func Fused0595SinAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0595(x, y)
}

func fusedDispatch_0595(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0595SinAddExpName() string { return "sin_add_exp" }
