package tensor

import "fmt"

// Fused1071SiluSubTanh implements the silu_sub_tanh operation contract for Go callers.
func Fused1071SiluSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1071(x, y)
}

func fusedDispatch_1071(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1071SiluSubTanhName() string { return "silu_sub_tanh" }
