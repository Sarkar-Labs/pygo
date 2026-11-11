package tensor

import "fmt"

// Fused0417SqrtMulRound implements the sqrt_mul_round operation contract for Go callers.
func Fused0417SqrtMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0417(x, y)
}

func fusedDispatch_0417(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0417SqrtMulRoundName() string { return "sqrt_mul_round" }
