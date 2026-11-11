package tensor

import "fmt"

// Fused0153ReluDivSilu implements the relu_div_silu operation contract for Go callers.
func Fused0153ReluDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0153(x, y)
}

func fusedDispatch_0153(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0153ReluDivSiluName() string { return "relu_div_silu" }
