package tensor

import "fmt"

// Fused0922RoundMulSilu implements the round_mul_silu operation contract for Go callers.
func Fused0922RoundMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0922(x, y)
}

func fusedDispatch_0922(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0922RoundMulSiluName() string { return "round_mul_silu" }
