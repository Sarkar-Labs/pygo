package tensor

import "fmt"

func Fused1108RoundDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Exp()
	return r, nil
}
func Fused1108RoundDivExpName() string { return "round_div_exp" }
