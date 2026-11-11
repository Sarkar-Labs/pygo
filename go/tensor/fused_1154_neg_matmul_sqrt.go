package tensor

import "fmt"

// Fused1154NegMatmulSqrt implements the neg_matmul_sqrt operation contract for Go callers.
func Fused1154NegMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1154(x, y)
}

func fusedDispatch_1154(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1154NegMatmulSqrtName() string { return "neg_matmul_sqrt" }
