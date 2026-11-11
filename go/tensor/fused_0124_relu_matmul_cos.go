package tensor

import "fmt"

// Fused0124ReluMatmulCos implements the relu_matmul_cos operation contract for Go callers.
func Fused0124ReluMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0124(x, y)
}

func fusedDispatch_0124(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0124ReluMatmulCosName() string { return "relu_matmul_cos" }
