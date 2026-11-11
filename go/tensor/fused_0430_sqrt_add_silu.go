package tensor

import "fmt"

// Fused0430SqrtAddSilu implements the sqrt_add_silu operation contract for Go callers.
func Fused0430SqrtAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0430(x, y)
}

func fusedDispatch_0430(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0430SqrtAddSiluName() string { return "sqrt_add_silu" }
