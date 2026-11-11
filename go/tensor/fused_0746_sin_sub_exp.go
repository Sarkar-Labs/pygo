package tensor

import "fmt"

func Fused0746SinSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Exp()
	return r, nil
}
func Fused0746SinSubExpName() string { return "sin_sub_exp" }
