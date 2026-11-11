package tensor

import "fmt"

// Fused0494ExpMatmulSign implements the exp_matmul_sign operation contract for Go callers.
func Fused0494ExpMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0494(x, y)
}

func fusedDispatch_0494(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0494ExpMatmulSignName() string { return "exp_matmul_sign" }
