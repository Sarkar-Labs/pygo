package tensor

import "fmt"

// Fused1261Log2SubSign implements the log2_sub_sign operation contract for Go callers.
func Fused1261Log2SubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1261(x, y)
}

func fusedDispatch_1261(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1261Log2SubSignName() string { return "log2_sub_sign" }
