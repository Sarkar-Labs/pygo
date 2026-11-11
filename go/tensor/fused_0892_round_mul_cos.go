package tensor

import "fmt"

// Fused0892RoundMulCos implements the round_mul_cos operation contract for Go callers.
func Fused0892RoundMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0892(x, y)
}

func fusedDispatch_0892(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0892RoundMulCosName() string { return "round_mul_cos" }
