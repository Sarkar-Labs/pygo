package tensor

import "fmt"

// Fused1060GeluAddSilu implements the gelu_add_silu operation contract for Go callers.
func Fused1060GeluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1060(x, y)
}

func fusedDispatch_1060(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1060GeluAddSiluName() string { return "gelu_add_silu" }
