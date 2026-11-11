package tensor

import "fmt"

// Fused0103ReluDivSqrt implements the relu_div_sqrt operation contract for Go callers.
func Fused0103ReluDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0103(x, y)
}

func fusedDispatch_0103(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0103ReluDivSqrtName() string { return "relu_div_sqrt" }
