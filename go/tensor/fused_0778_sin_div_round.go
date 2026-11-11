package tensor

import "fmt"

func Fused0778SinDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Round()
	return r, nil
}
func Fused0778SinDivRoundName() string { return "sin_div_round" }
