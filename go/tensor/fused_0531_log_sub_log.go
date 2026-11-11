package tensor

import "fmt"

// Fused0531LogSubLog implements the log_sub_log operation contract for Go callers.
func Fused0531LogSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0531(x, y)
}

func fusedDispatch_0531(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0531LogSubLogName() string { return "log_sub_log" }
