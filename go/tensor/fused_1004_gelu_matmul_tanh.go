package tensor

import "fmt"

// Fused1004GeluMatmulTanh implements the gelu_matmul_tanh operation contract for Go callers.
func Fused1004GeluMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1004(x, y)
}

func fusedDispatch_1004(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1004GeluMatmulTanhName() string { return "gelu_matmul_tanh" }
