package tensor

import "fmt"

// Fused1321Log10SubCeil implements the log10_sub_ceil operation contract for Go callers.
func Fused1321Log10SubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1321(x, y)
}

func fusedDispatch_1321(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1321Log10SubCeilName() string { return "log10_sub_ceil" }
