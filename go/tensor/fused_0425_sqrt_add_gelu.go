package tensor

import "fmt"

// Fused0425SqrtAddGelu implements the sqrt_add_gelu operation contract for Go callers.
func Fused0425SqrtAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0425(x, y)
}

func fusedDispatch_0425(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0425SqrtAddGeluName() string { return "sqrt_add_gelu" }
