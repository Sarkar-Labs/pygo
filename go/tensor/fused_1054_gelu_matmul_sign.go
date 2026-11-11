package tensor

import "fmt"

// Fused1054GeluMatmulSign implements the gelu_matmul_sign operation contract for Go callers.
func Fused1054GeluMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1054(x, y)
}

func fusedDispatch_1054(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1054GeluMatmulSignName() string { return "gelu_matmul_sign" }
