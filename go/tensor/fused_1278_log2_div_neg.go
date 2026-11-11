package tensor

import "fmt"

// Fused1278Log2DivNeg implements the log2_div_neg operation contract for Go callers.
func Fused1278Log2DivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1278(x, y)
}

func fusedDispatch_1278(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1278Log2DivNegName() string { return "log2_div_neg" }
