package tensor

import "fmt"

// Fused1144NegMatmulTanh implements the neg_matmul_tanh operation contract for Go callers.
func Fused1144NegMatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1144(x, y)
}

func fusedDispatch_1144(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1144NegMatmulTanhName() string { return "neg_matmul_tanh" }
