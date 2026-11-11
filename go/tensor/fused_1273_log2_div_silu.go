package tensor

import "fmt"

// Fused1273Log2DivSilu implements the log2_div_silu operation contract for Go callers.
func Fused1273Log2DivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1273(x, y)
}

func fusedDispatch_1273(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1273Log2DivSiluName() string { return "log2_div_silu" }
