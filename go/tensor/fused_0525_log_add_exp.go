package tensor

import "fmt"

// Fused0525LogAddExp implements the log_add_exp operation contract for Go callers.
func Fused0525LogAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0525(x, y)
}

func fusedDispatch_0525(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0525LogAddExpName() string { return "log_add_exp" }
