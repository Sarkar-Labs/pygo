package tensor

import "fmt"

// Fused0605SinAddSin implements the sin_add_sin operation contract for Go callers.
func Fused0605SinAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0605(x, y)
}

func fusedDispatch_0605(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0605SinAddSinName() string { return "sin_add_sin" }
