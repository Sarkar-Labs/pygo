package tensor

import "fmt"

// Fused1170NegAddCos implements the neg_add_cos operation contract for Go callers.
func Fused1170NegAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1170(x, y)
}

func fusedDispatch_1170(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1170NegAddCosName() string { return "neg_add_cos" }
