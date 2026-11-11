package tensor

import "fmt"

func Fused0728SinDivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Sin()
	r = r.Sigmoid()
	return r, nil
}
func Fused0728SinDivSigmoidName() string { return "sin_div_sigmoid" }
