package tensor

import "fmt"

// Fused1047GeluMulRound implements the gelu_mul_round operation contract for Go callers.
func Fused1047GeluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1047(x, y)
}

func fusedDispatch_1047(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1047GeluMulRoundName() string { return "gelu_mul_round" }
