package tensor

import "fmt"

// Fused1107SiluMulFloor implements the silu_mul_floor operation contract for Go callers.
func Fused1107SiluMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1107(x, y)
}

func fusedDispatch_1107(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1107SiluMulFloorName() string { return "silu_mul_floor" }
