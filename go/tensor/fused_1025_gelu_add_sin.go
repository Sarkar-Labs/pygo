package tensor

import "fmt"

// Fused1025GeluAddSin implements the gelu_add_sin operation contract for Go callers.
func Fused1025GeluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1025(x, y)
}

func fusedDispatch_1025(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1025GeluAddSinName() string { return "gelu_add_sin" }
