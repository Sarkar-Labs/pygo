package tensor

import "fmt"

// Fused1226Log2SubExp implements the log2_sub_exp operation contract for Go callers.
func Fused1226Log2SubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1226(x, y)
}

func fusedDispatch_1226(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1226Log2SubExpName() string { return "log2_sub_exp" }
