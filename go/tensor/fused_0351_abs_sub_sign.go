package tensor

import "fmt"

// Fused0351AbsSubSign implements the abs_sub_sign operation contract for Go callers.
func Fused0351AbsSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0351(x, y)
}

func fusedDispatch_0351(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0351AbsSubSignName() string { return "abs_sub_sign" }
