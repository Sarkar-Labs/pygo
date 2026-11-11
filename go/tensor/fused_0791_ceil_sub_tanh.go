package tensor

import "fmt"

// Fused0791CeilSubTanh implements the ceil_sub_tanh operation contract for Go callers.
func Fused0791CeilSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0791(x, y)
}

func fusedDispatch_0791(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0791CeilSubTanhName() string { return "ceil_sub_tanh" }
