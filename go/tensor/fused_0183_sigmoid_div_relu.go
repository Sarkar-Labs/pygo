package tensor

import "fmt"

func Fused0183SigmoidDivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.ReLU()
	return r, nil
}
func Fused0183SigmoidDivReluName() string { return "sigmoid_div_relu" }
