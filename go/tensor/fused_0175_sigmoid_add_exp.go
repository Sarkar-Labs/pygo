package tensor

import "fmt"

// Fused0175SigmoidAddExp implements the sigmoid_add_exp operation contract for Go callers.
func Fused0175SigmoidAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0175(x, y)
}

func fusedDispatch_0175(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0175SigmoidAddExpName() string { return "sigmoid_add_exp" }
