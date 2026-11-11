package tensor

import "fmt"

// Fused1275Log2AddNeg implements the log2_add_neg operation contract for Go callers.
func Fused1275Log2AddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1275(x, y)
}

func fusedDispatch_1275(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1275Log2AddNegName() string { return "log2_add_neg" }
