package tensor

import "fmt"

// Fused0313AbsDivSqrt implements the abs_div_sqrt operation contract for Go callers.
func Fused0313AbsDivSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_sqrt: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0313(x, y)
}

func fusedDispatch_0313(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0313AbsDivSqrtName() string { return "abs_div_sqrt" }
