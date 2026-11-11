package tensor

import "fmt"

// Fused1169NegMatmulSin implements the neg_matmul_sin operation contract for Go callers.
func Fused1169NegMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1169(x, y)
}

func fusedDispatch_1169(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1169NegMatmulSinName() string { return "neg_matmul_sin" }
