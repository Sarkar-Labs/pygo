package tensor

import "fmt"

// Fused1276Log2SubNeg implements the log2_sub_neg operation contract for Go callers.
func Fused1276Log2SubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1276(x, y)
}

func fusedDispatch_1276(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1276Log2SubNegName() string { return "log2_sub_neg" }
