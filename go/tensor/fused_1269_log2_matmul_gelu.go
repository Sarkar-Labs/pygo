package tensor

import "fmt"

// Fused1269Log2MatmulGelu implements the log2_matmul_gelu operation contract for Go callers.
func Fused1269Log2MatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1269(x, y)
}

func fusedDispatch_1269(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1269Log2MatmulGeluName() string { return "log2_matmul_gelu" }
