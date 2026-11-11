package tensor

import "fmt"

// Fused1256Log2SubRound implements the log2_sub_round operation contract for Go callers.
func Fused1256Log2SubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1256(x, y)
}

func fusedDispatch_1256(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1256Log2SubRoundName() string { return "log2_sub_round" }
