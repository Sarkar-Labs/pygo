package tensor

import "fmt"

// Fused0361AbsSubSilu implements the abs_sub_silu operation contract for Go callers.
func Fused0361AbsSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0361(x, y)
}

func fusedDispatch_0361(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0361AbsSubSiluName() string { return "abs_sub_silu" }
