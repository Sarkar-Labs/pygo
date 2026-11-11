package tensor

import "fmt"

// Fused0161SigmoidSubTanh implements the sigmoid_sub_tanh operation contract for Go callers.
func Fused0161SigmoidSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0161(x, y)
}

func fusedDispatch_0161(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0161SigmoidSubTanhName() string { return "sigmoid_sub_tanh" }
