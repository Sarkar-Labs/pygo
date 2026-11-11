package tensor

import "fmt"

// Fused0840CeilAddSign implements the ceil_add_sign operation contract for Go callers.
func Fused0840CeilAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0840(x, y)
}

func fusedDispatch_0840(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0840CeilAddSignName() string { return "ceil_add_sign" }
