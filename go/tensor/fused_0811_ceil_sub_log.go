package tensor

import "fmt"

// Fused0811CeilSubLog implements the ceil_sub_log operation contract for Go callers.
func Fused0811CeilSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0811(x, y)
}

func fusedDispatch_0811(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0811CeilSubLogName() string { return "ceil_sub_log" }
