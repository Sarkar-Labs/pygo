package tensor

import "fmt"

func Fused0868CosDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Round()
	return r, nil
}
func Fused0868CosDivRoundName() string { return "cos_div_round" }
