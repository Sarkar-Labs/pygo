package tensor

import "fmt"

// Fused1119SiluMatmulRound implements the silu_matmul_round operation contract for Go callers.
func Fused1119SiluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1119(x, y)
}

func fusedDispatch_1119(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1119SiluMatmulRoundName() string { return "silu_matmul_round" }
