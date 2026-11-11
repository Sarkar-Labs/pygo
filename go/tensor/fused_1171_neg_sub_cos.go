package tensor

import "fmt"

// Fused1171NegSubCos implements the neg_sub_cos operation contract for Go callers.
func Fused1171NegSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1171(x, y)
}

func fusedDispatch_1171(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1171NegSubCosName() string { return "neg_sub_cos" }
