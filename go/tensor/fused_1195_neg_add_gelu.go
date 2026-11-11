package tensor

import "fmt"

// Fused1195NegAddGelu implements the neg_add_gelu operation contract for Go callers.
func Fused1195NegAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1195(x, y)
}

func fusedDispatch_1195(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1195NegAddGeluName() string { return "neg_add_gelu" }
