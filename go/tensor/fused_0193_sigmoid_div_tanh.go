package tensor

import "fmt"

func Fused0193SigmoidDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.Sigmoid()
	r = r.Tanh()
	return r, nil
}
func Fused0193SigmoidDivTanhName() string { return "sigmoid_div_tanh" }
