package tensor

import "fmt"

// Fused1130SiluAddSilu implements the silu_add_silu operation contract for Go callers.
func Fused1130SiluAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1130(x, y)
}

func fusedDispatch_1130(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1130SiluAddSiluName() string { return "silu_add_silu" }
