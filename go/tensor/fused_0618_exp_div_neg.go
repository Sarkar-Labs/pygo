package tensor

import "fmt"

func Fused0618ExpDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Neg()
	return r, nil
}
func Fused0618ExpDivNegName() string { return "exp_div_neg" }
