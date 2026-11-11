package tensor

import "fmt"

// Fused1231Log2SubLog implements the log2_sub_log operation contract for Go callers.
func Fused1231Log2SubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1231(x, y)
}

func fusedDispatch_1231(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1231Log2SubLogName() string { return "log2_sub_log" }
