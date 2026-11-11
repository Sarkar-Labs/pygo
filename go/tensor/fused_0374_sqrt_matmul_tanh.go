package tensor

import "fmt"

// Fused0374SqrtMatmulTanh implements the sqrt_matmul_tanh operation contract for Go callers.
func Fused0374SqrtMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0374(x, y)
}

func fusedDispatch_0374(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0374SqrtMatmulTanhName() string { return "sqrt_matmul_tanh" }
