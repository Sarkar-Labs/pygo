package tensor

import "fmt"

func Fused0236SigmoidSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.Sigmoid()
	r = r.Round()
	return r, nil
}
func Fused0236SigmoidSubRoundName() string { return "sigmoid_sub_round" }
