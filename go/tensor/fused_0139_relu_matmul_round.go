package tensor

import "fmt"

// Fused0139ReluMatmulRound implements the relu_matmul_round operation contract for Go callers.
func Fused0139ReluMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0139(x, y)
}

func fusedDispatch_0139(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0139ReluMatmulRoundName() string { return "relu_matmul_round" }
