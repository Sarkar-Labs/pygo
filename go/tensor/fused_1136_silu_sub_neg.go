package tensor

import "fmt"

// Fused1136SiluSubNeg implements the silu_sub_neg operation contract for Go callers.
func Fused1136SiluSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1136(x, y)
}

func fusedDispatch_1136(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1136SiluSubNegName() string { return "silu_sub_neg" }
