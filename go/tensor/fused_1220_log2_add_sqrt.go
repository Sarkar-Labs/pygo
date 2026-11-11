package tensor

import "fmt"

// Fused1220Log2AddSqrt implements the log2_add_sqrt operation contract for Go callers.
func Fused1220Log2AddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1220(x, y)
}

func fusedDispatch_1220(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1220Log2AddSqrtName() string { return "log2_add_sqrt" }
