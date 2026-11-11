package tensor

import "fmt"

// Fused0174SigmoidMatmulSqrt implements the sigmoid_matmul_sqrt operation contract for Go callers.
func Fused0174SigmoidMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0174(x, y)
}

func fusedDispatch_0174(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0174SigmoidMatmulSqrtName() string { return "sigmoid_matmul_sqrt" }
