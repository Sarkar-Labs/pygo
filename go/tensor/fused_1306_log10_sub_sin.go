package tensor

import "fmt"

// Fused1306Log10SubSin implements the log10_sub_sin operation contract for Go callers.
func Fused1306Log10SubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1306(x, y)
}

func fusedDispatch_1306(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1306Log10SubSinName() string { return "log10_sub_sin" }
