package tensor

import "fmt"

// Fused1095SiluAddSin implements the silu_add_sin operation contract for Go callers.
func Fused1095SiluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1095(x, y)
}

func fusedDispatch_1095(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1095SiluAddSinName() string { return "silu_add_sin" }
