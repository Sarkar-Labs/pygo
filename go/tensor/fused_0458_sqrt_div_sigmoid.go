package tensor

import "fmt"

func Fused0458SqrtDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Sqrt()
	r = r.Sigmoid()
	return r, nil
}
func Fused0458SqrtDivSigmoidName() string { return "sqrt_div_sigmoid" }
