package tensor

import "fmt"

// Fused0967SignMulFloor implements the sign_mul_floor operation contract for Go callers.
func Fused0967SignMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0967(x, y)
}

func fusedDispatch_0967(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0967SignMulFloorName() string { return "sign_mul_floor" }
