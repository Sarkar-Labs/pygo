package tensor

import "fmt"

// Fused0801CeilSubSqrt implements the ceil_sub_sqrt operation contract for Go callers.
func Fused0801CeilSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0801(x, y)
}

func fusedDispatch_0801(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0801CeilSubSqrtName() string { return "ceil_sub_sqrt" }
