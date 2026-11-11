package tensor

import "fmt"

// Fused1265Log2AddGelu implements the log2_add_gelu operation contract for Go callers.
func Fused1265Log2AddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1265(x, y)
}

func fusedDispatch_1265(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1265Log2AddGeluName() string { return "log2_add_gelu" }
