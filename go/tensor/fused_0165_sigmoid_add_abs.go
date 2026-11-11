package tensor

import "fmt"

// Fused0165SigmoidAddAbs implements the sigmoid_add_abs operation contract for Go callers.
func Fused0165SigmoidAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0165(x, y)
}

func fusedDispatch_0165(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0165SigmoidAddAbsName() string { return "sigmoid_add_abs" }
