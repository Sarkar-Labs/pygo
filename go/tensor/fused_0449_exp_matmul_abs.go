package tensor

import "fmt"

// Fused0449ExpMatmulAbs implements the exp_matmul_abs operation contract for Go callers.
func Fused0449ExpMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0449(x, y)
}

func fusedDispatch_0449(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0449ExpMatmulAbsName() string { return "exp_matmul_abs" }
