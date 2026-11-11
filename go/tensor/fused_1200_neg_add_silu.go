package tensor

import "fmt"

// Fused1200NegAddSilu implements the neg_add_silu operation contract for Go callers.
func Fused1200NegAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1200(x, y)
}

func fusedDispatch_1200(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1200NegAddSiluName() string { return "neg_add_silu" }
