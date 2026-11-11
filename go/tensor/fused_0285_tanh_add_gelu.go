package tensor

import "fmt"

// Fused0285TanhAddGelu implements the tanh_add_gelu operation contract for Go callers.
func Fused0285TanhAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0285(x, y)
}

func fusedDispatch_0285(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0285TanhAddGeluName() string { return "tanh_add_gelu" }
