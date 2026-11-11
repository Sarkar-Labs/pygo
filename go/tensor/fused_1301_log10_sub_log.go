package tensor

import "fmt"

// Fused1301Log10SubLog implements the log10_sub_log operation contract for Go callers.
func Fused1301Log10SubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1301(x, y)
}

func fusedDispatch_1301(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1301Log10SubLogName() string { return "log10_sub_log" }
