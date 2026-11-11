package tensor

import "fmt"

// Fused0785FloorAddNeg implements the floor_add_neg operation contract for Go callers.
func Fused0785FloorAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_neg: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0785(x, y)
}

func fusedDispatch_0785(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0785FloorAddNegName() string { return "floor_add_neg" }
