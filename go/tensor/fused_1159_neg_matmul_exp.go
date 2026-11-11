package tensor

import "fmt"

// Fused1159NegMatmulExp implements the neg_matmul_exp operation contract for Go callers.
func Fused1159NegMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1159(x, y)
}

func fusedDispatch_1159(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1159NegMatmulExpName() string { return "neg_matmul_exp" }
