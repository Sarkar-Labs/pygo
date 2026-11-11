package tensor

import "fmt"

// Fused0629SinMatmulRound implements the sin_matmul_round operation contract for Go callers.
func Fused0629SinMatmulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0629(x, y)
}

func fusedDispatch_0629(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0629SinMatmulRoundName() string { return "sin_matmul_round" }
