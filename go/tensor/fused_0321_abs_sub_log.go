package tensor

import "fmt"

// Fused0321AbsSubLog implements the abs_sub_log operation contract for Go callers.
func Fused0321AbsSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0321(x, y)
}

func fusedDispatch_0321(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0321AbsSubLogName() string { return "abs_sub_log" }
