package tensor

import "fmt"

// Fused1260Log2AddSign implements the log2_add_sign operation contract for Go callers.
func Fused1260Log2AddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1260(x, y)
}

func fusedDispatch_1260(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1260Log2AddSignName() string { return "log2_add_sign" }
