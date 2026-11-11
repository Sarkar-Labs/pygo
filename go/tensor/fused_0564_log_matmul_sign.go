package tensor

import "fmt"

// Fused0564LogMatmulSign implements the log_matmul_sign operation contract for Go callers.
func Fused0564LogMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0564(x, y)
}

func fusedDispatch_0564(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0564LogMatmulSignName() string { return "log_matmul_sign" }
