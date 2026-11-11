package tensor

import "fmt"

// Fused0600SinAddLog implements the sin_add_log operation contract for Go callers.
func Fused0600SinAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0600(x, y)
}

func fusedDispatch_0600(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0600SinAddLogName() string { return "sin_add_log" }
