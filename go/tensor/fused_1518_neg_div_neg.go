package tensor

import "fmt"

func Fused1518NegDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Neg()
	return r, nil
}
func Fused1518NegDivNegName() string { return "neg_div_neg" }
