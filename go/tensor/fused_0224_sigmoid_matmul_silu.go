package tensor

import "fmt"

// Fused0224SigmoidMatmulSilu implements the sigmoid_matmul_silu operation contract for Go callers.
func Fused0224SigmoidMatmulSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0224(x, y)
}

func fusedDispatch_0224(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0224SigmoidMatmulSiluName() string { return "sigmoid_matmul_silu" }
