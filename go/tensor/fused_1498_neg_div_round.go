package tensor

import "fmt"

func Fused1498NegDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Round()
	return r, nil
}
func Fused1498NegDivRoundName() string { return "neg_div_round" }
