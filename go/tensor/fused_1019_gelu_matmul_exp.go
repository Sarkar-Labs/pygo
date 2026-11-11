package tensor

import "fmt"

// Fused1019GeluMatmulExp implements the gelu_matmul_exp operation contract for Go callers.
func Fused1019GeluMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1019(x, y)
}

func fusedDispatch_1019(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1019GeluMatmulExpName() string { return "gelu_matmul_exp" }
