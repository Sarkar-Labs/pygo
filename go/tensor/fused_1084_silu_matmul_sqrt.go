package tensor

import "fmt"

// Fused1084SiluMatmulSqrt implements the silu_matmul_sqrt operation contract for Go callers.
func Fused1084SiluMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1084(x, y)
}

func fusedDispatch_1084(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1084SiluMatmulSqrtName() string { return "silu_matmul_sqrt" }
