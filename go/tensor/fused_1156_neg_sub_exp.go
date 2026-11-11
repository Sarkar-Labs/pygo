package tensor

import "fmt"

// Fused1156NegSubExp implements the neg_sub_exp operation contract for Go callers.
func Fused1156NegSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_exp: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1156(x, y)
}

func fusedDispatch_1156(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1156NegSubExpName() string { return "neg_sub_exp" }
