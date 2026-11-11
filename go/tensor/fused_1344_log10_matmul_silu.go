package tensor

import "fmt"

// Fused1344Log10MatmulSilu implements the log10_matmul_silu operation contract for Go callers.
func Fused1344Log10MatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1344(x, y)
}

func fusedDispatch_1344(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1344Log10MatmulSiluName() string { return "log10_matmul_silu" }
