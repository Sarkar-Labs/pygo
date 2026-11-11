package tensor

import "fmt"

func Fused0103ReluDivTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_tanh: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Tanh()
	return r, nil
}
func Fused0103ReluDivTanhName() string { return "relu_div_tanh" }
