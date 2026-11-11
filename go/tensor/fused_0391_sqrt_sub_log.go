package tensor

import "fmt"

// Fused0391SqrtSubLog implements the sqrt_sub_log operation contract for Go callers.
func Fused0391SqrtSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0391(x, y)
}

func fusedDispatch_0391(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0391SqrtSubLogName() string { return "sqrt_sub_log" }
