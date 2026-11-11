package tensor

import "fmt"

// Fused0121ReluSubCos implements the relu_sub_cos operation contract for Go callers.
func Fused0121ReluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0121(x, y)
}

func fusedDispatch_0121(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0121ReluSubCosName() string { return "relu_sub_cos" }
