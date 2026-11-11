package tensor

import "fmt"

func Fused0548ExpDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Exp()
	r = r.Sigmoid()
	return r, nil
}
func Fused0548ExpDivSigmoidName() string { return "exp_div_sigmoid" }
