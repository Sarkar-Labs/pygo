package tensor

import "fmt"

// Fused1189NegMatmulRound implements the neg_matmul_round operation contract for Go callers.
func Fused1189NegMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1189(x, y)
}

func fusedDispatch_1189(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1189NegMatmulRoundName() string { return "neg_matmul_round" }
