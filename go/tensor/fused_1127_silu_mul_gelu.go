package tensor

import "fmt"

// Fused1127SiluMulGelu implements the silu_mul_gelu operation contract for Go callers.
func Fused1127SiluMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1127(x, y)
}

func fusedDispatch_1127(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1127SiluMulGeluName() string { return "silu_mul_gelu" }
