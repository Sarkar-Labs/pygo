package tensor

import "fmt"

func Fused0118ReluDivExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_exp: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Exp()
	return r, nil
}
func Fused0118ReluDivExpName() string { return "relu_div_exp" }
