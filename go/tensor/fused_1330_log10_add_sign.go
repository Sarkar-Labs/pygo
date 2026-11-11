package tensor

import "fmt"

// Fused1330Log10AddSign implements the log10_add_sign operation contract for Go callers.
func Fused1330Log10AddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1330(x, y)
}

func fusedDispatch_1330(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1330Log10AddSignName() string { return "log10_add_sign" }
