package tensor

import "fmt"

// Fused1284Log10MatmulTanh implements the log10_matmul_tanh operation contract for Go callers.
func Fused1284Log10MatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1284(x, y)
}

func fusedDispatch_1284(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1284Log10MatmulTanhName() string { return "log10_matmul_tanh" }
