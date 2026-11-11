package tensor

import "fmt"

func Fused0476SqrtSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Exp()
	return r, nil
}
func Fused0476SqrtSubExpName() string { return "sqrt_sub_exp" }
