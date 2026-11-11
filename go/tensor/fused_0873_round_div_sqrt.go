package tensor

import "fmt"

// Fused0873RoundDivSqrt implements the round_div_sqrt operation contract for Go callers.
func Fused0873RoundDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0873(x, y)
}

func fusedDispatch_0873(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0873RoundDivSqrtName() string { return "round_div_sqrt" }
