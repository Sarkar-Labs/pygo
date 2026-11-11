package tensor

import "fmt"

func Fused1077CeilMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Log10()
	return r, nil
}
func Fused1077CeilMulLog10Name() string { return "ceil_mul_log10" }
