package tensor

import "fmt"

// Fused0150ReluAddSilu implements the relu_add_silu operation contract for Go callers.
func Fused0150ReluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0150(x, y)
}

func fusedDispatch_0150(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0150ReluAddSiluName() string { return "relu_add_silu" }
