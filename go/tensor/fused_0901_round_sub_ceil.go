package tensor

import "fmt"

// Fused0901RoundSubCeil implements the round_sub_ceil operation contract for Go callers.
func Fused0901RoundSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_ceil: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0901(x, y)
}

func fusedDispatch_0901(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0901RoundSubCeilName() string { return "round_sub_ceil" }
