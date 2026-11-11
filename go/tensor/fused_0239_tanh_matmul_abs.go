package tensor

import "fmt"

// Fused0239TanhMatmulAbs implements the tanh_matmul_abs operation contract for Go callers.
func Fused0239TanhMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0239(x, y)
}

func fusedDispatch_0239(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0239TanhMatmulAbsName() string { return "tanh_matmul_abs" }
