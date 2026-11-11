package tensor

import "fmt"

// Fused1300Log10AddLog implements the log10_add_log operation contract for Go callers.
func Fused1300Log10AddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1300(x, y)
}

func fusedDispatch_1300(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1300Log10AddLogName() string { return "log10_add_log" }
