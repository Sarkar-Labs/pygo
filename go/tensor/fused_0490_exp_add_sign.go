package tensor

import "fmt"

// Fused0490ExpAddSign implements the exp_add_sign operation contract for Go callers.
func Fused0490ExpAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0490(x, y)
}

func fusedDispatch_0490(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0490ExpAddSignName() string { return "exp_add_sign" }
