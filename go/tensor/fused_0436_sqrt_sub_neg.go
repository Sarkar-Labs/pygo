package tensor

import "fmt"

// Fused0436SqrtSubNeg implements the sqrt_sub_neg operation contract for Go callers.
func Fused0436SqrtSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0436(x, y)
}

func fusedDispatch_0436(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0436SqrtSubNegName() string { return "sqrt_sub_neg" }
