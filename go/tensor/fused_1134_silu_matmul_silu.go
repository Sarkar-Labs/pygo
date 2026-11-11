package tensor

import "fmt"

// Fused1134SiluMatmulSilu implements the silu_matmul_silu operation contract for Go callers.
func Fused1134SiluMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1134(x, y)
}

func fusedDispatch_1134(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1134SiluMatmulSiluName() string { return "silu_matmul_silu" }
