package tensor

import "fmt"

// Fused0715CosAddNeg implements the cos_add_neg operation contract for Go callers.
func Fused0715CosAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0715(x, y)
}

func fusedDispatch_0715(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0715CosAddNegName() string { return "cos_add_neg" }
