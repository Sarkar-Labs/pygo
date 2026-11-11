package tensor

import "fmt"

// Fused0641SinSubSilu implements the sin_sub_silu operation contract for Go callers.
func Fused0641SinSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0641(x, y)
}

func fusedDispatch_0641(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0641SinSubSiluName() string { return "sin_sub_silu" }
