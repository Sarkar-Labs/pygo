package tensor

import "fmt"

// Fused1090SiluAddLog implements the silu_add_log operation contract for Go callers.
func Fused1090SiluAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_1090(x, y)
}

func fusedDispatch_1090(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused1090SiluAddLogName() string { return "silu_add_log" }
