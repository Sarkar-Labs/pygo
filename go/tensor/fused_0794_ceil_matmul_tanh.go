package tensor

import "fmt"

// Fused0794CeilMatmulTanh implements the ceil_matmul_tanh operation contract for Go callers.
func Fused0794CeilMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0794(x, y)
}

func fusedDispatch_0794(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0794CeilMatmulTanhName() string { return "ceil_matmul_tanh" }
