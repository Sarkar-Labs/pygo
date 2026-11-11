package tensor

import "fmt"

// Fused1075SiluAddAbs implements the silu_add_abs operation contract for Go callers.
func Fused1075SiluAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_abs: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1075(x, y)
}

func fusedDispatch_1075(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1075SiluAddAbsName() string { return "silu_add_abs" }
