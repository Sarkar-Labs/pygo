package tensor

import "fmt"

func Fused0616ExpSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.Neg()
	return r, nil
}
func Fused0616ExpSubNegName() string { return "exp_sub_neg" }
