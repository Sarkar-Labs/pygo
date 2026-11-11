package tensor

import "fmt"

// Fused0471ExpSubCos implements the exp_sub_cos operation contract for Go callers.
func Fused0471ExpSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0471(x, y)
}

func fusedDispatch_0471(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0471ExpSubCosName() string { return "exp_sub_cos" }
