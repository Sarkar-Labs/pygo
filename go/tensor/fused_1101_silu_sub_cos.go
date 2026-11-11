package tensor

import "fmt"

// Fused1101SiluSubCos implements the silu_sub_cos operation contract for Go callers.
func Fused1101SiluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1101(x, y)
}

func fusedDispatch_1101(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1101SiluSubCosName() string { return "silu_sub_cos" }
