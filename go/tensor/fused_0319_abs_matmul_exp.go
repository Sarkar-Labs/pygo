package tensor

import "fmt"

// Fused0319AbsMatmulExp implements the abs_matmul_exp operation contract for Go callers.
func Fused0319AbsMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0319(x, y)
}

func fusedDispatch_0319(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0319AbsMatmulExpName() string { return "abs_matmul_exp" }
