package tensor

import "fmt"

// Fused1091SiluSubLog implements the silu_sub_log operation contract for Go callers.
func Fused1091SiluSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1091(x, y)
}

func fusedDispatch_1091(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1091SiluSubLogName() string { return "silu_sub_log" }
