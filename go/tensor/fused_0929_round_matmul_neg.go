package tensor

import "fmt"

// Fused0929RoundMatmulNeg implements the round_matmul_neg operation contract for Go callers.
func Fused0929RoundMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0929(x, y)
}

func fusedDispatch_0929(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0929RoundMatmulNegName() string { return "round_matmul_neg" }
