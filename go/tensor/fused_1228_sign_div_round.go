package tensor

import "fmt"

func Fused1228SignDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Sign()
	r = r.Round()
	return r, nil
}
func Fused1228SignDivRoundName() string { return "sign_div_round" }
