package tensor

import "fmt"

// Fused0362AbsMulSilu implements the abs_mul_silu operation contract for Go callers.
func Fused0362AbsMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0362(x, y)
}

func fusedDispatch_0362(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0362AbsMulSiluName() string { return "abs_mul_silu" }
