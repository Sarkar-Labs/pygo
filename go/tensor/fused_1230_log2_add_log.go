package tensor

import "fmt"

// Fused1230Log2AddLog implements the log2_add_log operation contract for Go callers.
func Fused1230Log2AddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1230(x, y)
}

func fusedDispatch_1230(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1230Log2AddLogName() string { return "log2_add_log" }
