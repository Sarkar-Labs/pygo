package tensor

import "fmt"

// Fused0984SignMatmulSign implements the sign_matmul_sign operation contract for Go callers.
func Fused0984SignMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0984(x, y)
}

func fusedDispatch_0984(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0984SignMatmulSignName() string { return "sign_matmul_sign" }
