package tensor

import "fmt"

func Fused0148ReluDivRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_round: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Round()
	return r, nil
}
func Fused0148ReluDivRoundName() string { return "relu_div_round" }
