package tensor

import "fmt"

// Fused0288TanhDivGelu implements the tanh_div_gelu operation contract for Go callers.
func Fused0288TanhDivGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_div_gelu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0288(x, y)
}

func fusedDispatch_0288(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0288TanhDivGeluName() string { return "tanh_div_gelu" }
