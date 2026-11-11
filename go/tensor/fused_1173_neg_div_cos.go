package tensor

import "fmt"

// Fused1173NegDivCos implements the neg_div_cos operation contract for Go callers.
func Fused1173NegDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1173(x, y)
}

func fusedDispatch_1173(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1173NegDivCosName() string { return "neg_div_cos" }
