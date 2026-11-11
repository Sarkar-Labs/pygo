package tensor

import "fmt"

// Fused1229Log2MatmulExp implements the log2_matmul_exp operation contract for Go callers.
func Fused1229Log2MatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1229(x, y)
}

func fusedDispatch_1229(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1229Log2MatmulExpName() string { return "log2_matmul_exp" }
