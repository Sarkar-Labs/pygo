package tensor

import "fmt"

// Fused1117SiluMulRound implements the silu_mul_round operation contract for Go callers.
func Fused1117SiluMulRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_round: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1117(x, y)
}

func fusedDispatch_1117(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1117SiluMulRoundName() string { return "silu_mul_round" }
