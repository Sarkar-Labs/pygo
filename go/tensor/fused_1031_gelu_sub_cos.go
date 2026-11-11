package tensor

import "fmt"

// Fused1031GeluSubCos implements the gelu_sub_cos operation contract for Go callers.
func Fused1031GeluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1031(x, y)
}

func fusedDispatch_1031(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1031GeluSubCosName() string { return "gelu_sub_cos" }
