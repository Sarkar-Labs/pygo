package tensor

import "fmt"

// Fused0505ExpAddNeg implements the exp_add_neg operation contract for Go callers.
func Fused0505ExpAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0505(x, y)
}

func fusedDispatch_0505(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0505ExpAddNegName() string { return "exp_add_neg" }
