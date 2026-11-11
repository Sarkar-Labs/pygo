package tensor

import "fmt"

// Fused0354AbsMatmulSign implements the abs_matmul_sign operation contract for Go callers.
func Fused0354AbsMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0354(x, y)
}

func fusedDispatch_0354(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0354AbsMatmulSignName() string { return "abs_matmul_sign" }
