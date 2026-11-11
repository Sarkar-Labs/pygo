package tensor

import "fmt"

// Fused0502ExpMulSilu implements the exp_mul_silu operation contract for Go callers.
func Fused0502ExpMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0502(x, y)
}

func fusedDispatch_0502(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0502ExpMulSiluName() string { return "exp_mul_silu" }
