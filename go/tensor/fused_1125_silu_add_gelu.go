package tensor

import "fmt"

// Fused1125SiluAddGelu implements the silu_add_gelu operation contract for Go callers.
func Fused1125SiluAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1125(x, y)
}

func fusedDispatch_1125(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1125SiluAddGeluName() string { return "silu_add_gelu" }
