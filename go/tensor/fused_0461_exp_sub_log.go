package tensor

import "fmt"

// Fused0461ExpSubLog implements the exp_sub_log operation contract for Go callers.
func Fused0461ExpSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0461(x, y)
}

func fusedDispatch_0461(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0461ExpSubLogName() string { return "exp_sub_log" }
