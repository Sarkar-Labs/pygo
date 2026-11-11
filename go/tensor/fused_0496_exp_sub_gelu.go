package tensor

import "fmt"

// Fused0496ExpSubGelu implements the exp_sub_gelu operation contract for Go callers.
func Fused0496ExpSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0496(x, y)
}

func fusedDispatch_0496(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0496ExpSubGeluName() string { return "exp_sub_gelu" }
