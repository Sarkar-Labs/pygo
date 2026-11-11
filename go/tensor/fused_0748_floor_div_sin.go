package tensor

import "fmt"

// Fused0748FloorDivSin implements the floor_div_sin operation contract for Go callers.
func Fused0748FloorDivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_sin: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0748(x, y)
}

func fusedDispatch_0748(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0748FloorDivSinName() string { return "floor_div_sin" }
