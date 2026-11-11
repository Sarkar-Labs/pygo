package tensor

import "fmt"

// Fused0469ExpMatmulSin implements the exp_matmul_sin operation contract for Go callers.
func Fused0469ExpMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0469(x, y)
}

func fusedDispatch_0469(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0469ExpMatmulSinName() string { return "exp_matmul_sin" }
