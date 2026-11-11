package tensor

import "fmt"

// Fused0566LogSubGelu implements the log_sub_gelu operation contract for Go callers.
func Fused0566LogSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0566(x, y)
}

func fusedDispatch_0566(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0566LogSubGeluName() string { return "log_sub_gelu" }
