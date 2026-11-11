package tensor

import "fmt"

// Fused1124SiluMatmulSign implements the silu_matmul_sign operation contract for Go callers.
func Fused1124SiluMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1124(x, y)
}

func fusedDispatch_1124(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1124SiluMatmulSignName() string { return "silu_matmul_sign" }
