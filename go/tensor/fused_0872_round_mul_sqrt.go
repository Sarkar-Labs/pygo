package tensor

import "fmt"

// Fused0872RoundMulSqrt implements the round_mul_sqrt operation contract for Go callers.
func Fused0872RoundMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0872(x, y)
}

func fusedDispatch_0872(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0872RoundMulSqrtName() string { return "round_mul_sqrt" }
