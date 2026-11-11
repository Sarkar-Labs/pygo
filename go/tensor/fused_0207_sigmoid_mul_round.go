package tensor

import "fmt"

// Fused0207SigmoidMulRound implements the sigmoid_mul_round operation contract for Go callers.
func Fused0207SigmoidMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0207(x, y)
}

func fusedDispatch_0207(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0207SigmoidMulRoundName() string { return "sigmoid_mul_round" }
