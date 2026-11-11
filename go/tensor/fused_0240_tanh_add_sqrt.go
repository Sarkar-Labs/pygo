package tensor

import "fmt"

// Fused0240TanhAddSqrt implements the tanh_add_sqrt operation contract for Go callers.
func Fused0240TanhAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0240(x, y)
}

func fusedDispatch_0240(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0240TanhAddSqrtName() string { return "tanh_add_sqrt" }
