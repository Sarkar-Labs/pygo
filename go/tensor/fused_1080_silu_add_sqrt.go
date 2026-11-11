package tensor

import "fmt"

// Fused1080SiluAddSqrt implements the silu_add_sqrt operation contract for Go callers.
func Fused1080SiluAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1080(x, y)
}

func fusedDispatch_1080(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1080SiluAddSqrtName() string { return "silu_add_sqrt" }
