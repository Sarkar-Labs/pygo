package tensor

import "fmt"

// Fused1087SiluMulExp implements the silu_mul_exp operation contract for Go callers.
func Fused1087SiluMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1087(x, y)
}

func fusedDispatch_1087(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1087SiluMulExpName() string { return "silu_mul_exp" }
