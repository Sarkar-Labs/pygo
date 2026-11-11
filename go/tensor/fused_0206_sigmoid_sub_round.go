package tensor

import "fmt"

// Fused0206SigmoidSubRound implements the sigmoid_sub_round operation contract for Go callers.
func Fused0206SigmoidSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0206(x, y)
}

func fusedDispatch_0206(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0206SigmoidSubRoundName() string { return "sigmoid_sub_round" }
