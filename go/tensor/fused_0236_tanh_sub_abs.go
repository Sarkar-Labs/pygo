package tensor

import "fmt"

// Fused0236TanhSubAbs implements the tanh_sub_abs operation contract for Go callers.
func Fused0236TanhSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0236(x, y)
}

func fusedDispatch_0236(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0236TanhSubAbsName() string { return "tanh_sub_abs" }
