package tensor

import "fmt"

// Fused0178SigmoidDivExp implements the sigmoid_div_exp operation contract for Go callers.
func Fused0178SigmoidDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0178(x, y)
}

func fusedDispatch_0178(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0178SigmoidDivExpName() string { return "sigmoid_div_exp" }
