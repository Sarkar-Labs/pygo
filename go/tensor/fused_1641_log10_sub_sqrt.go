package tensor

import "fmt"

func Fused1641Log10SubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Sqrt()
	return r, nil
}
func Fused1641Log10SubSqrtName() string { return "log10_sub_sqrt" }
