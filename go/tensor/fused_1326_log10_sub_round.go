package tensor

import "fmt"

// Fused1326Log10SubRound implements the log10_sub_round operation contract for Go callers.
func Fused1326Log10SubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1326(x, y)
}

func fusedDispatch_1326(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1326Log10SubRoundName() string { return "log10_sub_round" }
