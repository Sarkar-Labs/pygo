package tensor

import "fmt"

func Fused0566ExpSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Exp()
	return r, nil
}
func Fused0566ExpSubExpName() string { return "exp_sub_exp" }
