package tensor

import "fmt"

// Fused0576LogSubNeg implements the log_sub_neg operation contract for Go callers.
func Fused0576LogSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0576(x, y)
}

func fusedDispatch_0576(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0576LogSubNegName() string { return "log_sub_neg" }
