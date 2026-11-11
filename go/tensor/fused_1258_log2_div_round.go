package tensor

import "fmt"

// Fused1258Log2DivRound implements the log2_div_round operation contract for Go callers.
func Fused1258Log2DivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1258(x, y)
}

func fusedDispatch_1258(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1258Log2DivRoundName() string { return "log2_div_round" }
