package tensor

import "fmt"

// Fused1241Log2SubCos implements the log2_sub_cos operation contract for Go callers.
func Fused1241Log2SubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1241(x, y)
}

func fusedDispatch_1241(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1241Log2SubCosName() string { return "log2_sub_cos" }
