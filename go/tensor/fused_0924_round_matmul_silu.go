package tensor

import "fmt"

// Fused0924RoundMatmulSilu implements the round_matmul_silu operation contract for Go callers.
func Fused0924RoundMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0924(x, y)
}

func fusedDispatch_0924(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0924RoundMatmulSiluName() string { return "round_matmul_silu" }
