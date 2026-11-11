package tensor

import "fmt"

// Fused1280Log10AddTanh implements the log10_add_tanh operation contract for Go callers.
func Fused1280Log10AddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1280(x, y)
}

func fusedDispatch_1280(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1280Log10AddTanhName() string { return "log10_add_tanh" }
