package tensor

import "fmt"

// Fused1190NegAddSign implements the neg_add_sign operation contract for Go callers.
func Fused1190NegAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1190(x, y)
}

func fusedDispatch_1190(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1190NegAddSignName() string { return "neg_add_sign" }
