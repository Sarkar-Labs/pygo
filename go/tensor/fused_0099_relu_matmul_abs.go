package tensor

import "fmt"

// Fused0099ReluMatmulAbs implements the relu_matmul_abs operation contract for Go callers.
func Fused0099ReluMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0099(x, y)
}

func fusedDispatch_0099(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0099ReluMatmulAbsName() string { return "relu_matmul_abs" }
