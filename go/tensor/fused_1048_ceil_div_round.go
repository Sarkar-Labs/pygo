package tensor

import "fmt"

func Fused1048CeilDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Round()
	return r, nil
}
func Fused1048CeilDivRoundName() string { return "ceil_div_round" }
