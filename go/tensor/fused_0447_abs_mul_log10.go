package tensor

import "fmt"

func Fused0447AbsMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Log10()
	return r, nil
}
func Fused0447AbsMulLog10Name() string { return "abs_mul_log10" }
