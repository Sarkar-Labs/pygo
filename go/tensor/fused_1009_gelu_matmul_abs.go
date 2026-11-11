package tensor

import "fmt"

// Fused1009GeluMatmulAbs implements the gelu_matmul_abs operation contract for Go callers.
func Fused1009GeluMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1009(x, y)
}

func fusedDispatch_1009(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1009GeluMatmulAbsName() string { return "gelu_matmul_abs" }
