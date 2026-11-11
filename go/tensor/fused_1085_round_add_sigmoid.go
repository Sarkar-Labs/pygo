package tensor

import "fmt"

func Fused1085RoundAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Sigmoid()
	return r, nil
}
func Fused1085RoundAddSigmoidName() string { return "round_add_sigmoid" }
