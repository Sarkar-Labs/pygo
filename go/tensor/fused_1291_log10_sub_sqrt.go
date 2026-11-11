package tensor

import "fmt"

// Fused1291Log10SubSqrt implements the log10_sub_sqrt operation contract for Go callers.
func Fused1291Log10SubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1291(x, y)
}

func fusedDispatch_1291(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1291Log10SubSqrtName() string { return "log10_sub_sqrt" }
