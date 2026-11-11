package tensor

import "fmt"

func Fused0418AbsDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Round()
	return r, nil
}
func Fused0418AbsDivRoundName() string { return "abs_div_round" }
