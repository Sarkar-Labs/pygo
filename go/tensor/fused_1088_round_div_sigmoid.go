package tensor

import "fmt"

func Fused1088RoundDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Round()
	r = r.Sigmoid()
	return r, nil
}
func Fused1088RoundDivSigmoidName() string { return "round_div_sigmoid" }
