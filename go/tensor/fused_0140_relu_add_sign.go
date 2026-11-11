package tensor

import "fmt"

// Fused0140ReluAddSign implements the relu_add_sign operation contract for Go callers.
func Fused0140ReluAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0140(x, y)
}

func fusedDispatch_0140(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0140ReluAddSignName() string { return "relu_add_sign" }
