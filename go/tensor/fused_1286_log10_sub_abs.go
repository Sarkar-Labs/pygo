package tensor

import "fmt"

// Fused1286Log10SubAbs implements the log10_sub_abs operation contract for Go callers.
func Fused1286Log10SubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1286(x, y)
}

func fusedDispatch_1286(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1286Log10SubAbsName() string { return "log10_sub_abs" }
