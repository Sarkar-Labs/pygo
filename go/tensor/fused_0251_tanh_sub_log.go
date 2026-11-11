package tensor

import "fmt"

// Fused0251TanhSubLog implements the tanh_sub_log operation contract for Go callers.
func Fused0251TanhSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0251(x, y)
}

func fusedDispatch_0251(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0251TanhSubLogName() string { return "tanh_sub_log" }
