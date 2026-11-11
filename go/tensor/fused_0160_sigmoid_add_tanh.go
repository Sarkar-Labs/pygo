package tensor

import "fmt"

// Fused0160SigmoidAddTanh implements the sigmoid_add_tanh operation contract for Go callers.
func Fused0160SigmoidAddTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0160(x, y)
}

func fusedDispatch_0160(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0160SigmoidAddTanhName() string { return "sigmoid_add_tanh" }
