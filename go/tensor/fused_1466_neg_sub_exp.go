package tensor

import "fmt"

func Fused1466NegSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.Exp()
	return r, nil
}
func Fused1466NegSubExpName() string { return "neg_sub_exp" }
