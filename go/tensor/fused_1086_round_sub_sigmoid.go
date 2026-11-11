package tensor

import "fmt"

func Fused1086RoundSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Sigmoid()
	return r, nil
}
func Fused1086RoundSubSigmoidName() string { return "round_sub_sigmoid" }
