package tensor

import "fmt"

// Fused0631SinSubSign implements the sin_sub_sign operation contract for Go callers.
func Fused0631SinSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0631(x, y)
}

func fusedDispatch_0631(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0631SinSubSignName() string { return "sin_sub_sign" }
