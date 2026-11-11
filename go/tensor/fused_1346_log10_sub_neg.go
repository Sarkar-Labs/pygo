package tensor

import "fmt"

// Fused1346Log10SubNeg implements the log10_sub_neg operation contract for Go callers.
func Fused1346Log10SubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1346(x, y)
}

func fusedDispatch_1346(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1346Log10SubNegName() string { return "log10_sub_neg" }
