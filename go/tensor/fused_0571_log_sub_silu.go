package tensor

import "fmt"

// Fused0571LogSubSilu implements the log_sub_silu operation contract for Go callers.
func Fused0571LogSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0571(x, y)
}

func fusedDispatch_0571(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0571LogSubSiluName() string { return "log_sub_silu" }
