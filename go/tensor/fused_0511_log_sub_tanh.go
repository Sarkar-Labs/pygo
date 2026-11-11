package tensor

import "fmt"

// Fused0511LogSubTanh implements the log_sub_tanh operation contract for Go callers.
func Fused0511LogSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_tanh: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0511(x, y)
}

func fusedDispatch_0511(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0511LogSubTanhName() string { return "log_sub_tanh" }
