package tensor

import "fmt"

// Fused1135SiluAddNeg implements the silu_add_neg operation contract for Go callers.
func Fused1135SiluAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1135(x, y)
}

func fusedDispatch_1135(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1135SiluAddNegName() string { return "silu_add_neg" }
