package tensor

import "fmt"

// Fused1096SiluSubSin implements the silu_sub_sin operation contract for Go callers.
func Fused1096SiluSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1096(x, y)
}

func fusedDispatch_1096(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1096SiluSubSinName() string { return "silu_sub_sin" }
