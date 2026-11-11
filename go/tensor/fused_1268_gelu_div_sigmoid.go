package tensor

import "fmt"

func Fused1268GeluDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.GELU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1268GeluDivSigmoidName() string { return "gelu_div_sigmoid" }
