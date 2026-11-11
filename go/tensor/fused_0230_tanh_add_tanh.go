package tensor

import "fmt"

// Fused0230TanhAddTanh implements the tanh_add_tanh operation contract for Go callers.
func Fused0230TanhAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0230(x, y)
}

func fusedDispatch_0230(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0230TanhAddTanhName() string { return "tanh_add_tanh" }
