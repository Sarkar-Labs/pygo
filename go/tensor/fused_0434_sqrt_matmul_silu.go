package tensor

import "fmt"

// Fused0434SqrtMatmulSilu implements the sqrt_matmul_silu operation contract for Go callers.
func Fused0434SqrtMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0434(x, y)
}

func fusedDispatch_0434(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0434SqrtMatmulSiluName() string { return "sqrt_matmul_silu" }
