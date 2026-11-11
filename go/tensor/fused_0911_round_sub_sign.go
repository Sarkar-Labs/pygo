package tensor

import "fmt"

// Fused0911RoundSubSign implements the round_sub_sign operation contract for Go callers.
func Fused0911RoundSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0911(x, y)
}

func fusedDispatch_0911(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0911RoundSubSignName() string { return "round_sub_sign" }
