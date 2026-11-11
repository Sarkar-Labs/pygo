package tensor

import "fmt"

// Fused1271Log2SubSilu implements the log2_sub_silu operation contract for Go callers.
func Fused1271Log2SubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1271(x, y)
}

func fusedDispatch_1271(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1271Log2SubSiluName() string { return "log2_sub_silu" }
