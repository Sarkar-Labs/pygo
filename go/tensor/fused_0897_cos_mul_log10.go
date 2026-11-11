package tensor

import "fmt"

func Fused0897CosMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Log10()
	return r, nil
}
func Fused0897CosMulLog10Name() string { return "cos_mul_log10" }
