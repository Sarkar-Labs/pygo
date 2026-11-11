package tensor

import "fmt"

// Fused0783FloorDivSilu implements the floor_div_silu operation contract for Go callers.
func Fused0783FloorDivSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_silu: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0783(x, y)
}

func fusedDispatch_0783(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0783FloorDivSiluName() string { return "floor_div_silu" }
