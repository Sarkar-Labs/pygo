package tensor

import "fmt"

func Fused1468NegDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Exp()
	return r, nil
}
func Fused1468NegDivExpName() string { return "neg_div_exp" }
