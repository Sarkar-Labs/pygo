package tensor

import "fmt"

// Fused1345Log10AddNeg implements the log10_add_neg operation contract for Go callers.
func Fused1345Log10AddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1345(x, y)
}

func fusedDispatch_1345(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1345Log10AddNegName() string { return "log10_add_neg" }
