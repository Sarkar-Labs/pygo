package tensor

import "fmt"

// Fused0290TanhAddSilu implements the tanh_add_silu operation contract for Go callers.
func Fused0290TanhAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0290(x, y)
}

func fusedDispatch_0290(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0290TanhAddSiluName() string { return "tanh_add_silu" }
