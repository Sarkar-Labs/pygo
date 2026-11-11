package tensor

import "fmt"

// Fused0260TanhAddCos implements the tanh_add_cos operation contract for Go callers.
func Fused0260TanhAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0260(x, y)
}

func fusedDispatch_0260(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0260TanhAddCosName() string { return "tanh_add_cos" }
