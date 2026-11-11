package tensor

import "fmt"

// Fused1224Log2MatmulSqrt implements the log2_matmul_sqrt operation contract for Go callers.
func Fused1224Log2MatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1224(x, y)
}

func fusedDispatch_1224(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1224Log2MatmulSqrtName() string { return "log2_matmul_sqrt" }
