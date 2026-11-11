package tensor

import "fmt"

func Fused1158RoundDivNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_neg: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Neg()
	return r, nil
}
func Fused1158RoundDivNegName() string { return "round_div_neg" }
