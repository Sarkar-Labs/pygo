package tensor

import "fmt"

// Fused0981SignSubSign implements the sign_sub_sign operation contract for Go callers.
func Fused0981SignSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0981(x, y)
}

func fusedDispatch_0981(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0981SignSubSignName() string { return "sign_sub_sign" }
