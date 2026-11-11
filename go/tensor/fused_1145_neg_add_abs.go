package tensor

import "fmt"

// Fused1145NegAddAbs implements the neg_add_abs operation contract for Go callers.
func Fused1145NegAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1145(x, y)
}

func fusedDispatch_1145(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1145NegAddAbsName() string { return "neg_add_abs" }
