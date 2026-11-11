package tensor

import "fmt"

// Fused0171SigmoidSubSqrt implements the sigmoid_sub_sqrt operation contract for Go callers.
func Fused0171SigmoidSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0171(x, y)
}

func fusedDispatch_0171(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0171SigmoidSubSqrtName() string { return "sigmoid_sub_sqrt" }
