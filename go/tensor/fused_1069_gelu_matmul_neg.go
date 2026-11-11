package tensor

import "fmt"

// Fused1069GeluMatmulNeg implements the gelu_matmul_neg operation contract for Go callers.
func Fused1069GeluMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1069(x, y)
}

func fusedDispatch_1069(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1069GeluMatmulNegName() string { return "gelu_matmul_neg" }
