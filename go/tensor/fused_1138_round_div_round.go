package tensor

import "fmt"

func Fused1138RoundDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Round()
	return r, nil
}
func Fused1138RoundDivRoundName() string { return "round_div_round" }
