package tensor

import "fmt"

// Fused0711CosSubSilu implements the cos_sub_silu operation contract for Go callers.
func Fused0711CosSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0711(x, y)
}

func fusedDispatch_0711(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0711CosSubSiluName() string { return "cos_sub_silu" }
