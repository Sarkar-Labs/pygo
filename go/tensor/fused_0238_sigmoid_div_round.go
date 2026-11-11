package tensor

import "fmt"

func Fused0238SigmoidDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Round()
	return r, nil
}
func Fused0238SigmoidDivRoundName() string { return "sigmoid_div_round" }
