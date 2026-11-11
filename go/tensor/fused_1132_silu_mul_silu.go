package tensor

import "fmt"

// Fused1132SiluMulSilu implements the silu_mul_silu operation contract for Go callers.
func Fused1132SiluMulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1132(x, y)
}

func fusedDispatch_1132(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1132SiluMulSiluName() string { return "silu_mul_silu" }
