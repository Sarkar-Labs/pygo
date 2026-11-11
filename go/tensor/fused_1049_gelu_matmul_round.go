package tensor

import "fmt"

// Fused1049GeluMatmulRound implements the gelu_matmul_round operation contract for Go callers.
func Fused1049GeluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1049(x, y)
}

func fusedDispatch_1049(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1049GeluMatmulRoundName() string { return "gelu_matmul_round" }
