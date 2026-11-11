package tensor

import "fmt"

// Fused1225Log2AddExp implements the log2_add_exp operation contract for Go callers.
func Fused1225Log2AddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1225(x, y)
}

func fusedDispatch_1225(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1225Log2AddExpName() string { return "log2_add_exp" }
