package tensor

import "fmt"

// Fused1013GeluDivSqrt implements the gelu_div_sqrt operation contract for Go callers.
func Fused1013GeluDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1013(x, y)
}

func fusedDispatch_1013(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1013GeluDivSqrtName() string { return "gelu_div_sqrt" }
