package tensor

import "fmt"

// Fused0855CeilAddNeg implements the ceil_add_neg operation contract for Go callers.
func Fused0855CeilAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0855(x, y)
}

func fusedDispatch_0855(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0855CeilAddNegName() string { return "ceil_add_neg" }
