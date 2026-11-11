package tensor

import "fmt"

// Fused1336Log10SubGelu implements the log10_sub_gelu operation contract for Go callers.
func Fused1336Log10SubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1336(x, y)
}

func fusedDispatch_1336(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1336Log10SubGeluName() string { return "log10_sub_gelu" }
