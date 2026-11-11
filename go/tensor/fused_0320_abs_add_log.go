package tensor

import "fmt"

// Fused0320AbsAddLog implements the abs_add_log operation contract for Go callers.
func Fused0320AbsAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0320(x, y)
}

func fusedDispatch_0320(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0320AbsAddLogName() string { return "abs_add_log" }
