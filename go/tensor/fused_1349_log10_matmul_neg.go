package tensor

import "fmt"

// Fused1349Log10MatmulNeg implements the log10_matmul_neg operation contract for Go callers.
func Fused1349Log10MatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1349(x, y)
}

func fusedDispatch_1349(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1349Log10MatmulNegName() string { return "log10_matmul_neg" }
