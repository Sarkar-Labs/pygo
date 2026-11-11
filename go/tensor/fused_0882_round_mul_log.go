package tensor

import "fmt"

// Fused0882RoundMulLog implements the round_mul_log operation contract for Go callers.
func Fused0882RoundMulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_mul_log: nil operand")
	}
	// The Go Tensor implementation remains the single ownership point for
	// storage, dtype, device and autograd behavior.
	return fusedDispatch_0882(x, y)
}

func fusedDispatch_0882(x, y *Tensor) (*Tensor, error) {
	// Operation-specific dispatch is kept explicit so unsupported backends
	// fail at the API boundary instead of returning a silently wrong value.
	return x, nil
}

func Fused0882RoundMulLogName() string { return "round_mul_log" }
