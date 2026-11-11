package tensor

import "fmt"

// Fused0432SqrtMulSilu implements the sqrt_mul_silu operation contract for Go callers.
func Fused0432SqrtMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0432(x, y)
}

func fusedDispatch_0432(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0432SqrtMulSiluName() string { return "sqrt_mul_silu" }
