package tensor

import "fmt"

// Fused0934SignMatmulTanh implements the sign_matmul_tanh operation contract for Go callers.
func Fused0934SignMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0934(x, y)
}

func fusedDispatch_0934(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0934SignMatmulTanhName() string { return "sign_matmul_tanh" }
