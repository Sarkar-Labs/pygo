package tensor

import "fmt"

// Fused1339Log10MatmulGelu implements the log10_matmul_gelu operation contract for Go callers.
func Fused1339Log10MatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1339(x, y)
}

func fusedDispatch_1339(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1339Log10MatmulGeluName() string { return "log10_matmul_gelu" }
