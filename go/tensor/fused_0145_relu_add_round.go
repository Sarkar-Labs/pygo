package tensor

import "fmt"

func Fused0145ReluAddRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_round: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Round()
	return r, nil
}
func Fused0145ReluAddRoundName() string { return "relu_add_round" }
