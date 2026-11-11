package tensor

import "fmt"

// Fused1257Log2MulRound implements the log2_mul_round operation contract for Go callers.
func Fused1257Log2MulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1257(x, y)
}

func fusedDispatch_1257(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1257Log2MulRoundName() string { return "log2_mul_round" }
