package tensor

import "fmt"

func Fused0146ReluSubRound(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_round: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Round()
	return r, nil
}
func Fused0146ReluSubRoundName() string { return "relu_sub_round" }
