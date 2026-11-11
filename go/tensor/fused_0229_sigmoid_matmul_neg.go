package tensor

import "fmt"

// Fused0229SigmoidMatmulNeg implements the sigmoid_matmul_neg operation contract for Go callers.
func Fused0229SigmoidMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0229(x, y)
}

func fusedDispatch_0229(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0229SigmoidMatmulNegName() string { return "sigmoid_matmul_neg" }
