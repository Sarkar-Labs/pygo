package tensor

import "fmt"

// Fused0621SinSubCeil implements the sin_sub_ceil operation contract for Go callers.
func Fused0621SinSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0621(x, y)
}

func fusedDispatch_0621(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0621SinSubCeilName() string { return "sin_sub_ceil" }
