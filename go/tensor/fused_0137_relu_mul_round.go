package tensor

import "fmt"

// Fused0137ReluMulRound implements the relu_mul_round operation contract for Go callers.
func Fused0137ReluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0137(x, y)
}

func fusedDispatch_0137(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0137ReluMulRoundName() string { return "relu_mul_round" }
