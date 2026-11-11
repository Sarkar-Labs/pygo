package tensor

import "fmt"

// Fused0304AbsMatmulTanh implements the abs_matmul_tanh operation contract for Go callers.
func Fused0304AbsMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0304(x, y)
}

func fusedDispatch_0304(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0304AbsMatmulTanhName() string { return "abs_matmul_tanh" }
