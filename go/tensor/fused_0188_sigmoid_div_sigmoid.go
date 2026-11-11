package tensor

import "fmt"

func Fused0188SigmoidDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Sigmoid()
	return r, nil
}
func Fused0188SigmoidDivSigmoidName() string { return "sigmoid_div_sigmoid" }
