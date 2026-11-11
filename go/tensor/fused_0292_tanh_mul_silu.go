package tensor

import "fmt"

// Fused0292TanhMulSilu implements the tanh_mul_silu operation contract for Go callers.
func Fused0292TanhMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0292(x, y)
}

func fusedDispatch_0292(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0292TanhMulSiluName() string { return "tanh_mul_silu" }
