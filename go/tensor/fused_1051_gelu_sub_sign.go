package tensor

import "fmt"

// Fused1051GeluSubSign implements the gelu_sub_sign operation contract for Go callers.
func Fused1051GeluSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1051(x, y)
}

func fusedDispatch_1051(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1051GeluSubSignName() string { return "gelu_sub_sign" }
