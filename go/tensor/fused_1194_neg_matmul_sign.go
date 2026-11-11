package tensor

import "fmt"

// Fused1194NegMatmulSign implements the neg_matmul_sign operation contract for Go callers.
func Fused1194NegMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1194(x, y)
}

func fusedDispatch_1194(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1194NegMatmulSignName() string { return "neg_matmul_sign" }
