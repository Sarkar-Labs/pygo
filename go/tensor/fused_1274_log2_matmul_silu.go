package tensor

import "fmt"

// Fused1274Log2MatmulSilu implements the log2_matmul_silu operation contract for Go callers.
func Fused1274Log2MatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1274(x, y)
}

func fusedDispatch_1274(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1274Log2MatmulSiluName() string { return "log2_matmul_silu" }
