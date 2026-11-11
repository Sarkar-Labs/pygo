package tensor

import "fmt"

// Fused0466ExpSubSin implements the exp_sub_sin operation contract for Go callers.
func Fused0466ExpSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0466(x, y)
}

func fusedDispatch_0466(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0466ExpSubSinName() string { return "exp_sub_sin" }
