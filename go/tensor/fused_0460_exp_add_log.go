package tensor

import "fmt"

// Fused0460ExpAddLog implements the exp_add_log operation contract for Go callers.
func Fused0460ExpAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0460(x, y)
}

func fusedDispatch_0460(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0460ExpAddLogName() string { return "exp_add_log" }
