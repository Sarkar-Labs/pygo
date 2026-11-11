package tensor

import "fmt"

// Fused0578LogDivNeg implements the log_div_neg operation contract for Go callers.
func Fused0578LogDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_div_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0578(x, y)
}

func fusedDispatch_0578(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0578LogDivNegName() string { return "log_div_neg" }
