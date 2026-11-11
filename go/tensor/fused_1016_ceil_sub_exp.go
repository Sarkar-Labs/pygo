package tensor

import "fmt"

func Fused1016CeilSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Exp()
	return r, nil
}
func Fused1016CeilSubExpName() string { return "ceil_sub_exp" }
