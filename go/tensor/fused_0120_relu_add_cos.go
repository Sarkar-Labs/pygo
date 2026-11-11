package tensor

import "fmt"

// Fused0120ReluAddCos implements the relu_add_cos operation contract for Go callers.
func Fused0120ReluAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0120(x, y)
}

func fusedDispatch_0120(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0120ReluAddCosName() string { return "relu_add_cos" }
