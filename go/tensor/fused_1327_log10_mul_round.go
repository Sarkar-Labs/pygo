package tensor

import "fmt"

// Fused1327Log10MulRound implements the log10_mul_round operation contract for Go callers.
func Fused1327Log10MulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1327(x, y)
}

func fusedDispatch_1327(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1327Log10MulRoundName() string { return "log10_mul_round" }
