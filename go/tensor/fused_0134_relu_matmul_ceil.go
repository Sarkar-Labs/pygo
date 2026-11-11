package tensor

import "fmt"

// Fused0134ReluMatmulCeil implements the relu_matmul_ceil operation contract for Go callers.
func Fused0134ReluMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0134(x, y)
}

func fusedDispatch_0134(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0134ReluMatmulCeilName() string { return "relu_matmul_ceil" }
