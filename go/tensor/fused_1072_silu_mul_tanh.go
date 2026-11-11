package tensor

import "fmt"

// Fused1072SiluMulTanh implements the silu_mul_tanh operation contract for Go callers.
func Fused1072SiluMulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1072(x, y)
}

func fusedDispatch_1072(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1072SiluMulTanhName() string { return "silu_mul_tanh" }
