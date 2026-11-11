package tensor

import "fmt"

func Fused1707Log10MulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.Log10()
	return r, nil
}
func Fused1707Log10MulLog10Name() string { return "log10_mul_log10" }
