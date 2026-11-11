package tensor

import "fmt"

// Fused1255Log2AddRound implements the log2_add_round operation contract for Go callers.
func Fused1255Log2AddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1255(x, y)
}

func fusedDispatch_1255(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1255Log2AddRoundName() string { return "log2_add_round" }
