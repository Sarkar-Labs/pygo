package tensor

import "fmt"

func Fused0008SigmoidDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	return r, nil
}
func Fused0008SigmoidDivName() string { return "sigmoid_div" }
