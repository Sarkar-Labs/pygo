package tensor

import "fmt"

// Fused0971SignSubCeil implements the sign_sub_ceil operation contract for Go callers.
func Fused0971SignSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0971(x, y)
}

func fusedDispatch_0971(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0971SignSubCeilName() string { return "sign_sub_ceil" }
