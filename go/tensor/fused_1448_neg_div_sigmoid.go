package tensor

import "fmt"

func Fused1448NegDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Neg()
	r = r.Sigmoid()
	return r, nil
}
func Fused1448NegDivSigmoidName() string { return "neg_div_sigmoid" }
