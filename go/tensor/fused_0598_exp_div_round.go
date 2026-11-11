package tensor

import "fmt"

func Fused0598ExpDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Round()
	return r, nil
}
func Fused0598ExpDivRoundName() string { return "exp_div_round" }
