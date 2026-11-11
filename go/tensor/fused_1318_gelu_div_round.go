package tensor

import "fmt"

func Fused1318GeluDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Round()
	return r, nil
}
func Fused1318GeluDivRoundName() string { return "gelu_div_round" }
