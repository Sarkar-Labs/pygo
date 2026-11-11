package tensor

import "fmt"

func Fused0615ExpAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.Neg()
	return r, nil
}
func Fused0615ExpAddNegName() string { return "exp_add_neg" }
