package tensor

import "fmt"

func Fused1408SiluDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.SiLU()
	r = r.Round()
	return r, nil
}
func Fused1408SiluDivRoundName() string { return "silu_div_round" }
