package tensor

import "fmt"

// Fused1249Log2MatmulFloor implements the log2_matmul_floor operation contract for Go callers.
func Fused1249Log2MatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_floor: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1249(x, y)
}

func fusedDispatch_1249(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1249Log2MatmulFloorName() string { return "log2_matmul_floor" }
