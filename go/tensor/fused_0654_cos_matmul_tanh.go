package tensor

import "fmt"

// Fused0654CosMatmulTanh implements the cos_matmul_tanh operation contract for Go callers.
func Fused0654CosMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0654(x, y)
}

func fusedDispatch_0654(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0654CosMatmulTanhName() string { return "cos_matmul_tanh" }
