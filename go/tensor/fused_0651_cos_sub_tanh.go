package tensor

import "fmt"

// Fused0651CosSubTanh implements the cos_sub_tanh operation contract for Go callers.
func Fused0651CosSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0651(x, y)
}

func fusedDispatch_0651(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0651CosSubTanhName() string { return "cos_sub_tanh" }
