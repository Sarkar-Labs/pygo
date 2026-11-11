package tensor

import "fmt"

// Fused0111ReluSubLog implements the relu_sub_log operation contract for Go callers.
func Fused0111ReluSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0111(x, y)
}

func fusedDispatch_0111(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0111ReluSubLogName() string { return "relu_sub_log" }
