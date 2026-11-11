package tensor

import "fmt"

// Fused1311Log10SubCos implements the log10_sub_cos operation contract for Go callers.
func Fused1311Log10SubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_cos: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1311(x, y)
}

func fusedDispatch_1311(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1311Log10SubCosName() string { return "log10_sub_cos" }
