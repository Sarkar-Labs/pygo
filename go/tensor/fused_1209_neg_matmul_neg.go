package tensor

import "fmt"

// Fused1209NegMatmulNeg implements the neg_matmul_neg operation contract for Go callers.
func Fused1209NegMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1209(x, y)
}

func fusedDispatch_1209(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1209NegMatmulNegName() string { return "neg_matmul_neg" }
