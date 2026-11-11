package tensor

import "fmt"

func Fused0818CosDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Sigmoid()
	return r, nil
}
func Fused0818CosDivSigmoidName() string { return "cos_div_sigmoid" }
