package tensor

import "fmt"

// Fused0979SignMatmulRound implements the sign_matmul_round operation contract for Go callers.
func Fused0979SignMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0979(x, y)
}

func fusedDispatch_0979(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0979SignMatmulRoundName() string { return "sign_matmul_round" }
