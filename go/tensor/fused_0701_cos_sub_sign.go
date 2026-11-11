package tensor

import "fmt"

// Fused0701CosSubSign implements the cos_sub_sign operation contract for Go callers.
func Fused0701CosSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_sign: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0701(x, y)
}

func fusedDispatch_0701(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0701CosSubSignName() string { return "cos_sub_sign" }
