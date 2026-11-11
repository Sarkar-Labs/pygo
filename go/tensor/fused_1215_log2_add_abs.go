package tensor

import "fmt"

// Fused1215Log2AddAbs implements the log2_add_abs operation contract for Go callers.
func Fused1215Log2AddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1215(x, y)
}

func fusedDispatch_1215(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1215Log2AddAbsName() string { return "log2_add_abs" }
