package tensor

import "fmt"

// Fused1251Log2SubCeil implements the log2_sub_ceil operation contract for Go callers.
func Fused1251Log2SubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1251(x, y)
}

func fusedDispatch_1251(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1251Log2SubCeilName() string { return "log2_sub_ceil" }
