package tensor

import "fmt"

func Fused0235SigmoidAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.Sigmoid()
	r = r.Round()
	return r, nil
}
func Fused0235SigmoidAddRoundName() string { return "sigmoid_add_round" }
