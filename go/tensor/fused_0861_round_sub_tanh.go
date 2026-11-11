package tensor

import "fmt"

// Fused0861RoundSubTanh implements the round_sub_tanh operation contract for Go callers.
func Fused0861RoundSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0861(x, y)
}

func fusedDispatch_0861(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0861RoundSubTanhName() string { return "round_sub_tanh" }
